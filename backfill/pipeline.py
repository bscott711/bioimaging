"""Per-dataset stage sequence for the bulk no-decon backfill.

Each function wraps exactly one registry stage (roi_detect, crop_zarr,
crop_tiff, deskew) and is individually resumable: a dataset interrupted
mid-backfill picks up at whichever stage the registry says isn't done yet,
never redoing already-completed work. No single dataset's exception is
allowed to propagate past `process_crop_and_submit` -- the bulk orchestrator
(backfill/cli.py) processes ~150-300 datasets and one bad dataset (e.g.
missing `_metadata.txt`) must not abort the rest.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path

import tifffile
import zarr
from opym.core import run_processing_job
from opym.discovery import LeafDataset, parse_zarr_group_prefix
from opym.metadata import parse_expected_timepoints, parse_z_step, resolve_zarr_z_step
from opym import lanes
from opym.petakit import resolve_deskew_working_dir, submit_remote_deskew_job
from opym.stream.live import live_status_is_fresh, read_live_status
from opym.registry import StatusRegistry, master_file_fingerprint
from opym.roi_detect import EXPECTED_H, EXPECTED_W, auto_detect_rois, compute_reference_projection
from opym.utils import (
    OutputFormat,
    derive_paths,
    resolve_output_base,
    scan_channel_patterns,
    write_decon_staged_tiff,
)
from psf_tools.extraction_plan import get_extraction_plan

from backfill.mip_movie import encode_poster_image, normalize_for_video

# Decon + DSR settings live in opym.decon_config, shared with the live lane
# (opym.stream.live) so both paths build identical tickets. The rationale for
# every value is documented there.
from opym.decon_config import (  # noqa: E402,F401 - re-exported for existing callers
    DECON_DAMP_FACTOR,
    DECON_EDGE_EROSION,
    DECON_HANN_WIN_BOUNDS,
    DECON_OTF_CUM_THRESH,
    DECON_WIENER_ALPHA,
    DSR_INTERP_METHOD,
    decon_params_fingerprint,
    deskew_decon_kwargs,
    dsr_dir_name_for,
    resolve_decon_psf,
)



class DatasetProcessingError(Exception):
    """Raised for a single dataset's stage failure. Always caught by the
    orchestrator -- never allowed to abort the rest of the backfill."""


class DeadDatasetError(Exception):
    """Raised when a Micro-Manager series' own master/base file -- not just
    a sibling continuation file -- has a corrupt (zero) TIFF header.
    tifffile's MMStack discovery always starts from the file it was asked
    to open, so there is no valid entry point to recover from; distinct
    from a plain read error so callers mark the dataset 'dead' (permanently
    unreadable) rather than 'corrupt' (recovery attempted, still failed) or
    a transient failure worth retrying.
    """


def _output_looks_present(path: Path) -> bool:
    """Cheap filesystem sanity check backing the registry's "done" status --
    protects against the registry saying done after someone manually deleted
    the output. Deliberately not an exact T*C file-count match (that needs
    re-opening the source file); existence + non-empty is enough to catch
    the common failure mode without adding real cost to every dataset.
    """
    return path.is_dir() and any(path.iterdir())


# Classic TIFF and BigTIFF magic numbers, both byte orders -- what a valid
# header actually starts with, regardless of what specific TIFF flavor a
# given acquisition file is.
_TIFF_MAGIC_BYTES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def _has_valid_tiff_header(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) in _TIFF_MAGIC_BYTES
    except OSError:
        return False


_SEQUENCE_SUFFIX_RE = re.compile(r"_(\d+)(?:\.ome)?\.tif$")


def _sequence_index(p: Path) -> int:
    """Where `p` falls in Micro-Manager's own rollover naming: the base
    file (`..._MMStack_Pos0.ome.tif`, no trailing `_N`) is 0, then
    `..._1.ome.tif`, `..._2.ome.tif`, ... in NUMERIC order. Plain
    lexicographic sorting would put `_10` before `_2`, which matters here:
    `resolve_readable_master` below walks this order to find the first
    corrupt file, so getting it wrong could silently include a file from
    past a real break as if it came before it.
    """
    m = _SEQUENCE_SUFFIX_RE.search(p.name)
    return int(m.group(1)) if m else 0


def _mmstack_sequence(master_file: Path) -> list[Path]:
    """Every file tifffile's own MMStack multi-file series discovery would
    consider part of the same acquisition as `master_file` -- the identical
    `<prefix>_MMStack*.tif` glob tifffile's `_series_mmstack` uses internally
    (`prefix = filename.split('_MMStack')[0]`), so this always agrees with
    what tifffile itself would try to open -- in true acquisition order (see
    `_sequence_index`), not the glob's arbitrary order. Not a Micro-Manager
    multi-file series at all (no `_MMStack` in the name) -> just the file
    itself.
    """
    if "_MMStack" not in master_file.name:
        return [master_file]
    prefix = master_file.name.split("_MMStack")[0]
    siblings = master_file.parent.glob(f"{prefix}_MMStack*.tif")
    return sorted(siblings, key=_sequence_index)


@dataclass(frozen=True)
class ReadableMaster:
    path: Path
    # None: nothing was excluded, trust the declared (T, Z, C, Y, X) shape.
    # An int: this many REAL (fully-written) timepoints were recovered --
    # fewer than the declared shape, which is zero-padded past this point.
    actual_timepoints: int | None
    excluded: tuple[Path, ...]


def resolve_readable_master(master_file: Path) -> ReadableMaster:
    """Confirmed real failure mode: a crashed Micro-Manager acquisition can
    leave one or more `*_MMStack_Pos0_N.ome.tif` sibling files with an
    all-zero header (either a trailing file that was created but never
    written, or a write that was truncated mid-flush) -- `tifffile` refuses
    to open the WHOLE multi-file series when ANY sibling fails its own TIFF
    magic check, even though the other siblings are perfectly good and
    hold real data.

    Builds a one-time, persistent mirror directory (hardlinked when
    possible -- same inode, instant, no extra disk use; a real copy only
    when that's not possible, e.g. across a filesystem boundary) containing
    just the valid siblings under the same name, so ordinary
    `tifffile.imread()`/`TiffFile()` -- unmodified, no special-casing
    anywhere else in this codebase -- sees a complete, uncorrupted series
    and applies its own normal multi-file logic. The corrupt file(s) are
    simply never linked into that directory, so tifffile's own glob never
    finds them; nothing here parses or reconstructs TIFF internals, which
    keeps this robust to corruption anywhere in the sequence (not just a
    trailing run) without having to reimplement tifffile's MMStack IFD
    stitching.

    Returns the ORIGINAL `master_file` unchanged (`actual_timepoints=None`)
    when every sibling already has a valid header -- the overwhelming
    common case, at the cost of one cheap 4-byte read per sibling.

    "Prefix" is enforced literally: this walks the sequence in acquisition
    order (`_mmstack_sequence`) and keeps only the leading run of good
    files, stopping at the FIRST corrupt one. A real dataset (`hDF_cell2`)
    confirmed this matters -- files `_4..._6` were corrupt but `_7..._10`
    afterward had perfectly valid headers. Micro-Manager rolls a long
    acquisition to a new file every ~4GB as it runs, so those later frames
    are NOT contiguous with the ones before the break; including them
    anyway would silently splice two separated time ranges together as if
    they were adjacent, which no caller here expects (`actual_timepoints`
    is used as a straight `[:N]` cap in `process_dataset`). Any good file
    after the first bad one is therefore excluded too, along with the bad
    one(s) -- less data recovered than a "keep every readable file" pass
    would manage, but never wrong about which timepoints are contiguous.

    Raises `DeadDatasetError` when `master_file` ITSELF is the corrupt one:
    tifffile's MMStack discovery always starts from the file it was asked
    to open, so there is no valid file to point it at instead.
    """
    sequence = _mmstack_sequence(master_file)

    good: list[Path] = []
    for p in sequence:
        if not _has_valid_tiff_header(p):
            break
        good.append(p)
    excluded = tuple(p for p in sequence if p not in good)

    if not excluded:
        return ReadableMaster(master_file, None, ())
    if not good or good[0] != master_file:
        raise DeadDatasetError(
            f"master file itself has a corrupt/zero TIFF header: {master_file}"
        )

    repair_dir = resolve_output_base(master_file.parent) / "_mmstack_valid_prefix"
    repair_dir.mkdir(parents=True, exist_ok=True)
    mirror_master = repair_dir / master_file.name
    for p in good:
        dest = repair_dir / p.name
        if dest.exists():
            continue
        try:
            os.link(p, dest)
        except OSError:
            # Cross-filesystem (raw data and the mirror root can be on
            # different mounts -- see resolve_output_base) or some other
            # reason hardlinking isn't possible here; a real copy is more
            # expensive but always works and only ever runs once per
            # dataset (the `dest.exists()` check above skips it on every
            # later pass).
            shutil.copy2(p, dest)

    # Count REAL (fully-written) timepoints from the valid files' own page
    # counts, independent of tifffile's own zero-fill behavior for a
    # missing MMStack file -- robust to the corrupt file(s) being anywhere
    # in the sequence, not just a trailing run.
    real_pages = 0
    for p in good:
        with tifffile.TiffFile(p) as tf:
            real_pages += len(tf.pages)

    actual_timepoints = None
    with tifffile.TiffFile(mirror_master) as tf:
        shape = tf.series[0].shape
    if len(shape) == 5:  # (T, Z, C, Y, X) -- a genuine multi-timepoint series
        per_timepoint_pages = 1
        for s in shape[1:-2]:  # every axis between T and the trailing (Y, X)
            per_timepoint_pages *= s
        if per_timepoint_pages:
            # Floor division: a final timepoint with fewer pages than a
            # full Z*C (partially written before the crash) is
            # conservatively excluded rather than counted as complete.
            actual_timepoints = real_pages // per_timepoint_pages

    return ReadableMaster(mirror_master, actual_timepoints, excluded)


def _open_lazy_zarr(master_file: Path) -> zarr.Array:
    store = tifffile.imread(str(master_file), aszarr=True)
    return zarr.open(store, mode="r")


def _clean_stale_deskew_output(dsr_dir: Path) -> None:
    """A retried deskew ticket resubmits against the same `dsr_dir` a prior
    failed attempt already wrote into -- confirmed via a real failed job
    log where PetaKit5D's `save('-v7.3', [dsrPath, '/parameters.mat'], pr)`
    failed with "Unable to write to file ... because it appears to be
    corrupt" against a half-written `parameters.mat` left behind by that
    earlier crash. Every subsequent retry hits the exact same corrupt-file
    error regardless of whether the underlying cause was fixed, since
    nothing ever clears the half-written state. Removing `dsr_dir` before
    resubmitting gives each retry a genuinely clean attempt.
    """
    if dsr_dir.exists():
        shutil.rmtree(dsr_dir)


def _channels_to_output(extraction_plan: list[tuple[int, bool, str, int]]) -> list[int] | None:
    channels = [out_c for (_raw_c, _is_top, _laser, out_c) in extraction_plan]
    return channels or None


def _fallback_centered_rois(
    shape: tuple[int, int], expected_h: int = EXPECTED_H, expected_w: int = EXPECTED_W
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """A centered, default-size box for each camera half, used when
    `auto_detect_rois` finds no signal at all. Keeps a likely-dud dataset
    flowing through crop/deskew/MIP (so it still gets a real thumbnail for
    visual triage) rather than aborting it outright -- `signal_flag`
    records the low-confidence classification separately, for
    prioritization, not as a hard failure.
    """
    max_y, max_x = shape
    half_y = max_y // 2

    def _centered(half_h: int, offset_y: int) -> tuple[slice, slice]:
        cy, cx = half_h // 2, max_x // 2
        y0 = max(0, cy - expected_h // 2)
        y1 = min(half_h, y0 + expected_h)
        x0 = max(0, cx - expected_w // 2)
        x1 = min(max_x, x0 + expected_w)
        return (slice(y0 + offset_y, y1 + offset_y), slice(x0, x1))

    return _centered(half_y, 0), _centered(max_y - half_y, half_y)


def _write_triage_preview(max_proj, out_path: Path) -> None:
    """Best-effort static preview JPEG from the raw (uncropped, undeskewed)
    reference max-projection -- gives every discovered dataset *something*
    visible in the dashboard immediately after `roi_detect`, well before
    crop/deskew/MIP (which can take much longer, or never happen at all for
    a deprioritized dud) finish.
    """
    try:
        preview_u8 = normalize_for_video(max_proj)
        encode_poster_image(preview_u8, out_path)
    except Exception as e:  # noqa: BLE001 - a missing preview is not fatal
        print(f"[backfill] triage preview failed for {out_path.parent.parent.name}: {e}")


def _classify_unreadable_raw_file(exc: Exception) -> str | None:
    """Returns `'corrupt'` iff `exc` indicates the raw file itself is
    fundamentally unreadable (truncated/corrupted acquisition, or corrupted
    TIFF metadata inside `tifffile`'s own parser), else `None` -- leaves the
    existing generic `failed` path untouched. Conservative by design: a
    false positive here would mislabel a real, fixable pipeline bug as
    unrecoverable data loss, so this only fires on confirmed signatures:

    - `tifffile.TiffFileError` ("not a TIFF file ...") -- the file's header
      isn't a valid TIFF magic number at all (confirmed via a real
      all-zero-header stub file from a crashed Micro-Manager acquisition).
    - a bare `IndexError` raised from WITHIN `tifffile`'s own source (its
      Micro-Manager multi-file series parser reads `databytecounts[0]` on
      what should be a per-page byte-offset tuple; a corrupted IFD tag
      makes that tuple empty) -- checked via the traceback's own frames so
      an unrelated `IndexError` bug in OUR code is never mislabeled as
      corrupt raw data.
    """
    if isinstance(exc, tifffile.TiffFileError):
        return "corrupt"
    if isinstance(exc, IndexError):
        if any(
            "tifffile" in (frame.filename or "")
            for frame in traceback.extract_tb(exc.__traceback__)
        ):
            return "corrupt"
    return None


def detect_rois(
    ds: LeafDataset, registry: StatusRegistry
) -> tuple[tuple | None, tuple | None, str]:
    """Always re-runs (cheap: one lazy plane read + a 2D gaussian filter/
    regionprops pass) rather than gating behind a registry skip-check --
    `master_roi.json`'s cached box size makes this deterministic/idempotent,
    and re-deriving the actual (slice, slice) ROI values is far cheaper than
    persisting and reloading them separately.

    Also records triage info used to prioritize the bulk run (see
    `backfill/cli.py`): a `signal_flag` ('ok' if either half has detectable
    signal, 'dud' if neither does) and an expected-vs-actual timepoint count
    (catches acquisitions that were configured for e.g. 100 timepoints but
    aborted after 1). Returns `(top_roi, bot_roi, signal_flag)` -- a 'dud'
    still gets a usable (centered default-size) ROI pair, so it still flows
    through crop/deskew/MIP, just at lower priority.

    Skips the real detection once `mip_encode` is already `done`: at that
    point `crop_and_convert`/`submit_deskew_ticket` are registry-gated
    no-ops that ignore whatever ROI this returns, so re-reading the raw
    file to re-derive it on every watch pass (this call happens twice per
    pass -- once from `backfill/cli.py`'s Phase 0 triage, once again here
    from Phase A) was pure waste. Confirmed live: a handful of large,
    single-timepoint calibration datasets re-triaged this way every 120s
    were slow/heavy enough (a full-stack projection, see
    `compute_reference_projection`) to occupy the entire worker pool and
    starve real pending datasets.

    Also skips a dataset already known 'dead' or 'corrupt' when the raw
    master file's (size, mtime) fingerprint hasn't changed since that
    triage was recorded: without this, a permanently unreadable raw file
    (a crashed acquisition with no recoverable data, or a bad IFD tag deep
    in the file that `resolve_readable_master` can't repair) gets re-read
    over NFS and re-fails IDENTICALLY on every single --watch pass forever.
    A re-upload/repair changes the fingerprint and is retried normally.
    """
    if registry.is_stage_done(ds.dataset_key, "mip_encode"):
        return None, None, "unknown"

    existing = registry.get_dataset(ds.dataset_key)
    if existing and existing.get("signal_flag") in ("dead", "corrupt"):
        current_fp = master_file_fingerprint(ds.master_file)
        if current_fp is not None and current_fp == existing.get("master_file_fingerprint"):
            return None, None, existing["signal_flag"]

    registry.start_stage(ds.dataset_key, "roi_detect")
    try:
        readable = resolve_readable_master(ds.master_file)
        z = _open_lazy_zarr(readable.path)
        master_roi_path = resolve_output_base(ds.leaf_dir) / "master_roi.json"
        # z.shape[0] is only the T axis for a genuine 5D (T,C,Z,Y,X)/
        # (T,Z,C,Y,X) array. A dataset with just one real timepoint arrives
        # here already squeezed to 4D (Z,C,Y,X) -- passing timepoint=None
        # in that case does a true full-stack max-projection; passing
        # z.shape[0]//2 (as if axis 0 were still T) would instead slice out
        # a single arbitrary Z-plane, an under-counting bug that could
        # misclassify real signal as a dud.
        timepoint = z.shape[0] // 2 if z.ndim >= 5 else None
        max_proj = compute_reference_projection(z, timepoint=timepoint)
        top_roi, bot_roi = auto_detect_rois(max_proj, master_roi_path=master_roi_path)

        signal_flag = "ok" if (top_roi is not None or bot_roi is not None) else "dud"
        if signal_flag == "dud":
            top_roi, bot_roi = _fallback_centered_rois(max_proj.shape)
        if readable.excluded and signal_flag == "ok":
            # Real data, just fewer timepoints than configured -- distinct
            # from 'dud' (readable, no signal at all) so the dashboard can
            # tell "partial recovery" from "nothing here worth processing".
            signal_flag = "partial"

        actual_timepoints = z.shape[0] if z.ndim >= 5 else 1
        if readable.actual_timepoints is not None:
            actual_timepoints = readable.actual_timepoints
        metadata_file = derive_paths(ds.master_file, OutputFormat.ZARR).metadata_file
        expected_timepoints = parse_expected_timepoints(metadata_file)
        registry.set_triage(
            ds.dataset_key,
            signal_flag=signal_flag,
            expected_timepoints=expected_timepoints,
            actual_timepoints=actual_timepoints,
        )
        if readable.excluded:
            print(
                f"[backfill] {ds.dataset_key}: excluded {len(readable.excluded)} corrupt "
                f"MMStack file(s) ({', '.join(p.name for p in readable.excluded)}), "
                f"recovered {actual_timepoints} real timepoint(s)"
            )
        _write_triage_preview(
            max_proj, resolve_output_base(ds.leaf_dir) / "mip_movies" / "triage_preview.jpg"
        )

        registry.finish_stage(ds.dataset_key, "roi_detect", status="done")
        return top_roi, bot_roi, signal_flag
    except DeadDatasetError as e:
        # set_signal_flag, not set_triage: this dataset may already have a
        # real expected/actual_timepoints from an earlier successful pass
        # (unlikely for a freshly-dead file, but not impossible -- e.g. a
        # previously-fine file corrupted later) -- set_triage would
        # silently null both out on every retry.
        registry.set_signal_flag(ds.dataset_key, "dead")
        registry.set_master_file_fingerprint(
            ds.dataset_key, master_file_fingerprint(ds.master_file)
        )
        registry.finish_stage(ds.dataset_key, "roi_detect", status="failed", error=str(e))
        raise
    except Exception as e:  # noqa: BLE001 - reported into the registry, then re-raised
        classification = _classify_unreadable_raw_file(e)
        if classification is not None:
            registry.set_signal_flag(ds.dataset_key, classification)
            registry.set_master_file_fingerprint(
                ds.dataset_key, master_file_fingerprint(ds.master_file)
            )
        registry.finish_stage(ds.dataset_key, "roi_detect", status="failed", error=str(e))
        raise


def crop_and_convert(
    ds: LeafDataset,
    top_roi: tuple | None,
    bot_roi: tuple | None,
    registry: StatusRegistry,
) -> tuple[Path, Path]:
    """Runs the existing `run_processing_job()` TWICE, unmodified:
    - OutputFormat.ZARR -> `processed_ngff/<name>_processed.zarr`, the
      durable crop + channel-remap deliverable ("that will become the raw
      data" for downstream analysis).
    - OutputFormat.TIFF_SERIES -> `processed_tiff_series_split/`, required
      because PetaKit5D's deskew-rotate dispatch only ever reads TIFF files,
      never zarr (confirmed directly against run_petakit_server.m).

    Both calls read the same lazy `aszarr()` view of the same raw file
    cropped to the same ROI, so the incremental cost of the second call is
    bounded by the (already-cropped) output size, not the full raw file.

    Reads through `resolve_readable_master(ds.master_file)` rather than
    `ds.master_file` directly, so a dataset with a corrupt MMStack sibling
    (already repaired once by `detect_rois`) reads from that same repaired
    mirror here too, capped to its real (non-zero-padded) timepoint count --
    see `ReadableMaster`/`process_dataset`'s `max_timepoints`. OUTPUT paths
    below stay derived from `ds.master_file` unchanged: the repaired mirror
    lives in its own `_mmstack_valid_prefix/` directory, unrelated to where
    this dataset's real output belongs.
    """
    readable = resolve_readable_master(ds.master_file)
    extraction_plan = get_extraction_plan(ds.master_file)
    channels_to_output = _channels_to_output(extraction_plan)

    # Derived through derive_paths(), not restated here, so this always
    # agrees with wherever run_processing_job() (below) actually writes --
    # including its fallback to a mirror location when ds.leaf_dir isn't
    # writable (see resolve_output_base).
    zarr_out_dir = derive_paths(ds.master_file, OutputFormat.ZARR).output_dir
    tiff_out_dir = derive_paths(ds.master_file, OutputFormat.TIFF_SERIES).output_dir

    if registry.is_stage_done(ds.dataset_key, "crop_zarr") and _output_looks_present(zarr_out_dir):
        pass
    else:
        registry.start_stage(ds.dataset_key, "crop_zarr")
        try:
            run_processing_job(
                base_file=ds.master_file,
                top_roi=top_roi,
                bottom_roi=bot_roi,
                output_format=OutputFormat.ZARR,
                channels_to_output=channels_to_output,
                rotate_90=True,
                # zarr_out_dir.parent is output_base (resolve_output_base's
                # result) -- keep this log beside the actual output, not
                # necessarily ds.leaf_dir, for the same reason zarr_out_dir
                # itself was moved off derive_paths' old hand-rolled form.
                cli_log_file=zarr_out_dir.parent / "opm_roi_log.json",
                read_file=readable.path,
                max_timepoints=readable.actual_timepoints,
            )
            registry.finish_stage(
                ds.dataset_key, "crop_zarr", status="done", output_path=str(zarr_out_dir)
            )
        except Exception as e:  # noqa: BLE001
            registry.finish_stage(ds.dataset_key, "crop_zarr", status="failed", error=str(e))
            raise DatasetProcessingError(f"crop_zarr failed: {e}") from e

    if registry.is_stage_done(ds.dataset_key, "crop_tiff") and _output_looks_present(tiff_out_dir):
        pass
    else:
        registry.start_stage(ds.dataset_key, "crop_tiff")
        try:
            run_processing_job(
                base_file=ds.master_file,
                top_roi=top_roi,
                bottom_roi=bot_roi,
                output_format=OutputFormat.TIFF_SERIES,
                channels_to_output=channels_to_output,
                rotate_90=True,
                # zarr_out_dir.parent is output_base (resolve_output_base's
                # result) -- keep this log beside the actual output, not
                # necessarily ds.leaf_dir, for the same reason zarr_out_dir
                # itself was moved off derive_paths' old hand-rolled form.
                cli_log_file=zarr_out_dir.parent / "opm_roi_log.json",
                read_file=readable.path,
                max_timepoints=readable.actual_timepoints,
            )
            registry.finish_stage(
                ds.dataset_key, "crop_tiff", status="done", output_path=str(tiff_out_dir)
            )
        except Exception as e:  # noqa: BLE001
            registry.finish_stage(ds.dataset_key, "crop_tiff", status="failed", error=str(e))
            raise DatasetProcessingError(f"crop_tiff failed: {e}") from e

    return zarr_out_dir, tiff_out_dir


def backfill_max_inflight() -> int | None:
    """OPYM_BACKFILL_MAX_INFLIGHT caps how many backfill tickets may be queued
    or running at once (opym-backfill.service sets it). Unset means no cap, so
    a manual one-shot run behaves exactly as it always has."""
    raw = os.environ.get("OPYM_BACKFILL_MAX_INFLIGHT", "").strip()
    return int(raw) if raw else None


def _backfill_lane_closed(ds: LeafDataset) -> bool:
    """Cheap pre-check, before any cleanup or staging: a live acquisition
    holds the GPUs, or the backfill already has its cap of tickets queued.
    The dataset is simply picked up again on a later pass."""
    if lanes.backfill_admission_open(backfill_max_inflight()):
        return False
    print(f"[backfill] {ds.dataset_key}: deferred (live acquisition or backfill queue full)")
    return True


def _admitted_submit(**kwargs) -> Path | None:
    """`submit_remote_deskew_job` under the backfill admission lock, so parallel
    Phase A workers can't all see room and overshoot the cap together. None
    if the lane closed since the pre-check."""
    with lanes.backfill_admission(backfill_max_inflight()) as admitted:
        if not admitted:
            return None
        return submit_remote_deskew_job(**kwargs)


def submit_deskew_ticket(
    ds: LeafDataset, tiff_out_dir: Path, registry: StatusRegistry
) -> Path | None:
    """Submits a deskew+rotate+MIP ticket via the *existing*
    `submit_remote_deskew_job`, optionally deconvolving first. `psf_path` is
    the entire decon switch (the ticket's `run_decon` server-side default is
    `~isempty(psf_path)`); `resolve_decon_psf()` returns None unless
    `--decon-psf` / `OPYM_DECON_PSF` is set, so the default is deskew-only,
    exactly as before. Routes through the 'deskew' job type, NOT
    `submit_pipeline_job`'s fused GPU 'pipeline' route, which has no
    skip-decon toggle.

    Unlike the zarr path, this needs no staging step: the crop stage already
    wrote its TIFFs in the rot90'd `(ny, nx, nz)` layout decon requires (and
    that the measured PSF itself carries). See `build_decon_staging_dir`.

    Returns the ticket path if one is pending resolution (freshly submitted,
    or already submitted by a prior run and not yet resolved), or None if
    this stage is already fully done -- the orchestrator's Phase B poller
    uses the returned path to know what to watch; None means "nothing to
    watch, already finished."
    """
    decon_psf = resolve_decon_psf()
    if registry.is_stage_done(ds.dataset_key, "deskew") and decon_provenance_matches(
        registry, ds.dataset_key, decon_psf
    ):
        return None

    existing = registry.get_stage(ds.dataset_key, "deskew")
    # Reuse an in-flight ticket only when it is for the configuration we
    # want. Provenance is recorded at SUBMIT time (see set_decon_psf below),
    # so a genuinely running job already matches; a `running` row left over
    # from an interrupted deskew-only pass does not, and would otherwise pin
    # the dataset to a ticket that will never produce decon output.
    if (
        existing
        and existing["status"] == "running"
        and existing["ticket_path"]
        and decon_provenance_matches(registry, ds.dataset_key, decon_psf)
    ):
        return Path(existing["ticket_path"])
    if _backfill_lane_closed(ds):
        return None
    # Past both early returns, so we are definitely submitting -- which means
    # whatever is on disk was made by a *different* configuration than the one
    # we are about to run (or by a run that crashed), and must not be left for
    # PetaKit5D to reuse. See submit_zarr_deskew_ticket for the full reasoning;
    # the gate used to be `status == "failed"`, which missed the case that
    # actually broke production: a parameter retune re-submitting over `done`.
    #
    # Requires an existing row, unlike the zarr path. `work_dir` here is the
    # crop stage's output dir, which is also where the older manual `opym` CLI
    # wrote its own `Decon/` -- so with no row at all we cannot tell this
    # pipeline's stale output from a hand-run result that predates it, and
    # deleting the latter is not ours to do. A row means this pipeline has
    # submitted for this dataset before and owns what is there.
    if existing:
        try:
            work_dir = resolve_deskew_working_dir(ds.master_file)
            _clean_stale_deskew_output(dsr_output_dir(work_dir, decon_psf))
            _clean_stale_deskew_output(work_dir / "Decon")
        except FileNotFoundError:
            pass

    paths = derive_paths(ds.master_file, OutputFormat.ZARR)  # only used for its metadata_file path
    z_step_um = parse_z_step(paths.metadata_file, default_z_step=0.3)
    channel_patterns_str = scan_channel_patterns(tiff_out_dir)
    channel_patterns = channel_patterns_str.split(", ") if channel_patterns_str else None

    ticket_path = _admitted_submit(
        input_target=ds.master_file,
        z_step_um=z_step_um,
        deskew=True,
        rotate=True,
        psf_path=decon_psf,
        channel_patterns=channel_patterns,
        save_mip=True,
        **deskew_decon_kwargs(decon_psf),
    )
    if ticket_path is None:
        return None
    registry.set_decon_psf(ds.dataset_key, str(decon_psf) if decon_psf else None)
    registry.set_decon_params(ds.dataset_key, decon_params_fingerprint() if decon_psf else None)
    registry.start_stage(ds.dataset_key, "deskew", ticket_path=str(ticket_path))
    # The MIPs on disk were made from the output this ticket replaces. A
    # `done` row left behind made opym-dashboard report ~450 re-submitted
    # datasets as finished, showing their old movies, for days.
    registry.reset_stage(ds.dataset_key, "mip_encode")
    return ticket_path


def _read_ome_zarr_dataset_path(store: Path, default: str = "p0") -> str:
    """Reads the OME-NGFF dataset path (e.g. "p0", "0") out of a zarr
    store's own `.zattrs` multiscales metadata, rather than assuming one
    fixed name -- different OME-Zarr writers use different level-0 path
    conventions.
    """
    try:
        attrs = json.loads((store / ".zattrs").read_text())
        return attrs["multiscales"][0]["datasets"][0]["path"]
    except Exception:  # noqa: BLE001 - malformed/unexpected attrs, fall back
        return default


def _zarr_store_is_ready(store: Path) -> bool:
    """True iff `store`'s OME-NGFF metadata AND its main pixel-data array
    are actually present -- confirmed via a real in-flight Globus transfer
    that a store's `.zgroup` (already required by
    `opym.discovery.is_zarr_leaf_dataset_dir`) and a SUBSET of its arrays
    (e.g. the `p`/`x`/`y`/`z` per-axis coordinate arrays this acquisition
    writer also stores) can land well before `.zattrs` and the main pixel
    array (`p0`) do -- `.zgroup` alone is not proof the store is complete.
    A store caught mid-transfer is simply not ready yet, not corrupt --
    the caller should skip it for this run and let the next run pick it up
    once Globus finishes, not record a hard failure.
    """
    if not (store / ".zattrs").is_file():
        return False
    dataset_path = _read_ome_zarr_dataset_path(store)
    return (store / dataset_path).is_dir()


def _read_zarray(pixel_dir: Path) -> dict:
    return json.loads((pixel_dir / ".zarray").read_text())


def channel_store_timepoints(store: Path) -> int:
    """Number of *written* timepoints on a per-channel zarr store's pixel
    array: 1 for a 3D `(z, y, x)` store, else the number of per-timepoint
    chunk directories present on a 4D `(t, z, y, x)` one (the newer
    pymmcore MDA writer's live-imaging output -- e.g. every macropinocytosis
    dataset).

    The `.zarray` shape is what PetaKit5D reads, but it reports the
    *declared* length -- what the acquisition was configured to collect, not
    what it finished. An aborted acquisition leaves a store declaring
    `t=100` with only `p0/0/` .. `p0/78/` on disk (real example: one cell in
    the 20260902 upload has 80 written on one channel and 79 on the other).
    Trusting the declared length made `build_zarr_pyramid_mirror` try to
    mirror a chunk directory that does not exist, which raises
    FileNotFoundError and takes the whole dataset's submission down with it.
    """
    pixel_dir = store / _read_ome_zarr_dataset_path(store)
    shape = _read_zarray(pixel_dir)["shape"]
    if len(shape) < 4:
        return 1
    written = sum(1 for e in pixel_dir.iterdir() if e.name.isdigit() and e.is_dir())
    return min(int(shape[0]), written)


def dataset_declared_timepoints(ds: LeafDataset) -> int:
    """How many timepoints a KIND_ZARR_PRECROPPED acquisition was configured
    to collect: the `.zarray` T length (see `channel_store_timepoints` for
    why that is the declared, not the written, count). `max` across
    channels, since each store is created with the full declared shape.
    Compared against `dataset_timepoints` so an aborted acquisition shows up
    as e.g. 2/100 instead of looking complete.
    """
    declared = []
    for store in ds.channel_zarr_paths:
        shape = _read_zarray(store / _read_ome_zarr_dataset_path(store))["shape"]
        declared.append(int(shape[0]) if len(shape) >= 4 else 1)
    return max(declared, default=1)


def dataset_timepoints(ds: LeafDataset) -> int:
    """Timepoint count for a KIND_ZARR_PRECROPPED dataset -- the `min`
    across its channel stores.

    `min`, not `max`: a timepoint is only usable once every channel has
    written it, and an interrupted acquisition genuinely stops mid-timepoint
    with one channel a frame ahead of the other. Taking the max would mirror
    a timepoint that one channel never wrote.
    """
    per_channel = [channel_store_timepoints(s) for s in ds.channel_zarr_paths]
    if not per_channel:
        return 1
    if len(set(per_channel)) > 1:
        print(
            f"   [{ds.dataset_key}] channels disagree on timepoint count "
            f"{per_channel}; using {min(per_channel)} (acquisition likely "
            "interrupted mid-timepoint)"
        )
    return min(per_channel)


def build_zarr_pyramid_mirror(
    channel_zarr_paths: tuple[Path, ...],
    mirror_dir: Path,
    *,
    dataset_prefix: str | None = None,
    max_timepoints: int | None = None,
) -> Path:
    """Per-dataset symlink mirror that presents this acquisition's OME-NGFF
    zarr stores to PetaKit5D as plain flat zarr v2 arrays -- the one shape
    all three of its independent zarr readers actually handle:

    - `parallelReadZarr` (the compiled mex `readzarr.m` tries FIRST): reads
      `<dir>/.zarray` then opens each chunk at `<dir>/<chunk-subfolder>/.../
      <chunk file>` directly. On a chunk file it can't open it silently
      skips it (`cpp-zarr/src/parallelreadzarr.cpp` ~line 103) -- by design,
      so a sparse array reads as its fill value -- and never raises, so
      `readzarr.m`'s `ZarrAdapter` fallback never runs. A mirror that put
      the real chunks anywhere but directly under `<dir>` therefore read as
      **all zeros with no error** (the original blank-DSR bug).
    - `ZarrAdapter.openToRead` (the fallback): `py.zarr.open(<dir>)` when
      there's no `<dir>/.zgroup`. A flat `.zarray` array opens fine; a
      `.zgroup` would send it looking for an `L_1_1_1/` pyramid level with
      its own `.zarray`, which an OME store doesn't have (pixels live in
      `p0/`, and a per-timepoint slice of `p0/` has no `.zarray` at all).
    - `getImageSize.m`: `fopen`s `<dir>/.zarray` unconditionally.

    So each mirrored store is exactly: a `.zarray` (symlinked from the real
    `p0/.zarray`, or written when it's a 3D view of a 4D store) plus one
    symlink per real chunk-index subfolder. No `.zgroup`, no `L_1_1_1`.

    A 4D `(t, z, y, x)` time-series store is exploded into one 3D mirrored
    store per timepoint (`<prefix>_C<ch>_T<ttt>.zarr`), because PetaKit5D's
    deskew/rotate path (`XR_deskewRotateFrame` -> `deskewFrame3D` /
    `rotateFrame3D`) is strictly 3D. Its multi-timepoint model, same as the
    legacy TIFF-series path (`opym.core`'s `{name}_C{c}_T{t:03d}.tif`), is
    "many 3D files discovered by `channelPatterns`", not "one 4D array".
    `dataset_prefix` names those per-T stores (defaults to the shared
    channel-name prefix); `max_timepoints` caps the count (cheap test runs).

    Symlink-only, read-only: no pixel data is copied and the real synced
    acquisition data is never touched -- every link lives under `mirror_dir`,
    this pipeline's own per-dataset output namespace.
    """
    def _relink(link_path: Path, target: Path) -> None:
        """`Path.exists()` follows symlinks and reports False for a
        DANGLING one (e.g. built while the store was still mid-Globus-
        transfer, before its real target existed) -- so a plain
        `if not link_path.exists()` guard would try to recreate an already-
        present dangling link and crash with FileExistsError (confirmed via
        a real run against a store still mid-transfer). Check `is_symlink()`
        instead, which reports the link's own presence regardless of target
        validity, and always repoint it to `target` so a mirror built
        against incomplete data self-heals once the real data lands.
        """
        if link_path.is_symlink() or link_path.exists():
            if link_path.resolve() == target:
                return
            link_path.unlink()
        link_path.symlink_to(target)

    def _mirror_store_dir(dst: Path, chunk_src: Path, zarray: Path | str) -> None:
        """One flat mirrored zarr array: `.zarray` (a Path to symlink, or a
        JSON string to write for a 3D view of a 4D store) plus one symlink
        per real chunk-index subfolder directly under `dst` -- where every
        PetaKit5D zarr reader looks for chunks.
        """
        dst.mkdir(exist_ok=True)
        za = dst / ".zarray"
        if isinstance(zarray, Path):
            _relink(za, zarray.resolve())
        else:
            if za.is_symlink():
                za.unlink()
            za.write_text(zarray)
        keep = {".zarray"}
        for entry in chunk_src.iterdir():
            if entry.name.startswith("."):
                continue
            _relink(dst / entry.name, entry.resolve())
            keep.add(entry.name)
        # Drop stale links from a previous build -- an old `.zgroup`/
        # `L_1_1_1` from the pre-flat-array layout, or chunk subfolders that
        # no longer belong to this view -- so `parallelReadZarr` can't pick
        # up a chunk that isn't part of the array `.zarray` describes.
        for entry in dst.iterdir():
            if entry.is_symlink() and entry.name not in keep:
                entry.unlink()

    mirror_dir.mkdir(parents=True, exist_ok=True)
    built: set[Path] = set()
    for cidx, store in enumerate(channel_zarr_paths):
        if not _zarr_store_is_ready(store):
            raise FileNotFoundError(
                f"{store} is missing .zattrs or its main pixel-data array -- "
                "likely still mid-Globus-transfer, will retry on next run"
            )
        pixel_dir = store / _read_ome_zarr_dataset_path(store)
        zarray = _read_zarray(pixel_dir)
        shape = zarray["shape"]

        if len(shape) < 4:
            # Already-3D store: `.zarray` symlinked straight through -- its
            # shape already matches what PetaKit5D should read.
            built.add(mirror_dir / store.name)
            _mirror_store_dir(mirror_dir / store.name, pixel_dir, pixel_dir / ".zarray")
            continue

        # Single-vs-multi naming is decided from `channel_store_timepoints`
        # (REAL written chunk dirs), not the store's raw declared `shape[0]`
        # and not the loop bound `n_t` below (which `max_timepoints` can
        # clamp for an intentionally-fast test run on an otherwise-healthy
        # store that must still get multi-style naming) -- see the
        # identical fix (and its full docstring) in `build_decon_staging_dir`.
        if channel_store_timepoints(store) <= 1:
            # Degenerate 4D (single real timepoint): still needs the same
            # leading-axis-stripped 3D view as the true time-series branch
            # below -- confirmed live that leaving `.zarray` as 4D
            # (T=1,Z,Y,X) and symlinking `pixel_dir` itself (whose chunk
            # keys are then genuinely 4-deep, `<T>/<Z>/<Y>/<X>`) produces a
            # self-consistent 4D array that PetaKit5D's C++ zarr reader --
            # strictly 3D -- can't read: it derives an expected per-chunk
            # byte size from the wrong 3 of the 4 chunk-shape entries, then
            # fails to decompress the real (differently-sized) chunk with a
            # generic "Decompression error. Error code: 0".
            #
            # Mirror directory naming intentionally stays `store.name` (NOT
            # the `_C{c}_T000` movie convention used below) -- unlike a
            # real time series, `_run_mip_encode`'s single-timepoint
            # ("poster") branch matches PetaKit5D's MIP output against
            # `channel_zarr_paths` names, not per-timepoint names.
            zarray_3d = json.dumps({**zarray, "shape": shape[1:], "chunks": zarray["chunks"][1:]})
            built.add(mirror_dir / store.name)
            _mirror_store_dir(mirror_dir / store.name, pixel_dir / "0", zarray_3d)
            continue

        # 4D time series -> one 3D mirrored store per timepoint. The 3D
        # `.zarray` is the 4D one minus its leading (t) axis; chunk data for
        # timepoint t lives under `pixel_dir/<t>/`.
        n_t = shape[0] if max_timepoints is None else min(shape[0], max_timepoints)
        zarray_3d = json.dumps({**zarray, "shape": shape[1:], "chunks": zarray["chunks"][1:]})
        prefix = dataset_prefix or parse_zarr_group_prefix(store)
        for t in range(n_t):
            chunk_src = pixel_dir / str(t)
            if not chunk_src.is_dir():
                # Belt and braces: callers already clamp `max_timepoints` to
                # what was written, but a store declaring more timepoints
                # than it holds must not take the whole dataset down with a
                # FileNotFoundError out of `_mirror_store_dir`.
                print(f"   Skipping T{t:03d} of {store.name}: {chunk_src} absent")
                continue
            dst = mirror_dir / f"{prefix}_C{cidx}_T{t:03d}.zarr"
            built.add(dst)
            _mirror_store_dir(dst, chunk_src, zarray_3d)
    # Drop per-timepoint stores left over from an earlier build. Without
    # this, a mirror built when the acquisition had written 69 timepoints
    # keeps its T068 store after a rebuild clamped to 68 -- PetaKit5D still
    # matches it on `_C0_T`, deskews it, and the channel counts diverge
    # (69 vs 68), which is what broke Cell_002's mip_encode with
    # "operands could not be broadcast together". `_mirror_store_dir` only
    # prunes *within* a store, never whole stale stores.
    for entry in mirror_dir.glob("*.zarr"):
        if entry not in built and entry.is_dir():
            print(f"   Removing stale mirror store {entry.name}")
            shutil.rmtree(entry)
    return mirror_dir


def dsr_output_dir(data_dir: Path, psf: Path | None) -> Path:
    """Where PetaKit5D actually writes the DSR result, given the ticket's
    `dataDir`.

    PetaKit5D writes DS/DSR *inside* whatever directory it was handed as
    `dataDir`. When decon runs first, the deskew step is not handed the
    ticket's `dataDir` -- `run_petakit_server.m` sets
    `current_input_dir = fullfile(job.dataDir, 'Decon')` and passes that
    instead -- so the DSR output nests one level deeper, under `Decon/`.

    Confirmed against a real completed job: with decon on, the output landed
    at `<dataDir>/Decon/DSR_decon`, not `<dataDir>/DSR_decon`. Getting this
    wrong is the same "No MIP TIFFs found on output that exists one directory
    over" failure `_dsr_dir_for` has already hit twice.
    """
    base = data_dir / "Decon" if psf else data_dir
    return base / dsr_dir_name_for(psf)


def reprocess_legacy_decon() -> bool:
    """True when datasets deconvolved before the `decon_params` fingerprint
    existed should be re-deconvolved with the current settings.

    Off by default: `decon_provenance_matches` otherwise reads a NULL
    fingerprint as "made with unknown, therefore wrong, parameters" and
    re-submits every legacy dataset at once. Read from the environment rather
    than threaded through as an argument for the same reason as
    `resolve_decon_psf` -- the backfill fans datasets across a process pool
    and an env var is inherited by every worker. `run_backfill_cli.py
    --reprocess-legacy-decon` sets it.
    """
    return os.environ.get("OPYM_DECON_REPROCESS_LEGACY", "").strip() not in ("", "0")


def log_grandfathered_decon_datasets(registry, psf: Path | None) -> int:
    """Print how many datasets are being held back by the NULL-fingerprint
    grandfather clause in `decon_provenance_matches`, and return the count.

    Exists so the exemption is visible once per pass instead of silent: a
    dataset that is skipped for a reason other than "already up to date" is
    exactly the thing this module keeps getting bitten by.
    """
    if psf is None or reprocess_legacy_decon():
        return 0
    # Only datasets the clause actually SKIPS count. A NULL fingerprint on a
    # dataset that never finished its deskew changes nothing -- it is not done,
    # so it re-submits on provenance or not -- and counting those inflated this
    # from the 65 datasets that really are being held to 539, which reads as a
    # far bigger exemption than it is.
    held = sum(
        1
        for d in registry.all_datasets()
        if d.get("decon_psf")
        and d.get("decon_params") is None
        and registry.is_stage_done(d["dataset_key"], "deskew")
    )
    if held:
        print(
            f"[backfill] {held} dataset(s) already deconvolved with unknown "
            "(pre-provenance) OMW settings are being left as-is -- their output "
            "will NOT be recomputed with the current ones. Rerun with "
            "--reprocess-legacy-decon (OPYM_DECON_REPROCESS_LEGACY=1) to "
            "re-deconvolve them."
        )
    return held


def decon_provenance_matches(registry, dataset_key: str, psf: Path | None) -> bool:
    """True when the output already on disk was made with both the PSF AND
    the OMW parameter settings (see `decon_params_fingerprint`) we are about
    to use.

    `is_stage_done(..., "deskew")` alone is not enough to skip a dataset: a
    stage is only "done" for the PSF *and settings* it was done WITH.
    Switching decon on for a corpus that was deskewed without it, changing
    PSF, or retuning alpha/OTFCumThresh/hann/damp while keeping the same PSF
    file, otherwise all look like a no-op, because every dataset reports
    itself already complete and nothing recomputes. That is the same
    silent-skip failure mode as PetaKit5D's own `if exist(deconFullpath,
    'file')`.
    """
    recorded_psf = registry.get_decon_psf(dataset_key)
    if recorded_psf != (str(psf) if psf else None):
        return False
    if psf is None:
        return True  # deskew-only: no OMW parameters were ever in play
    recorded_params = registry.get_decon_params(dataset_key)
    if recorded_params is None and not reprocess_legacy_decon():
        # Grandfathered: this dataset was deconvolved before `decon_params`
        # existed as a column, so we know the PSF matches but genuinely
        # cannot tell which OMW knobs produced it -- in practice the
        # pre-super4 settings (alpha 0.02). Treating "unknown" as "stale"
        # queues the entire legacy corpus (65 datasets as of 2026-09-21) for
        # re-deconvolution the instant the backfill restarts, which is a
        # corpus-wide GPU run nobody asked for. Hold them until someone opts
        # in deliberately and can batch it.
        #
        # Note this is exactly the silent-no-op this function exists to
        # prevent, deliberately reintroduced for one closed cohort -- hence
        # `log_grandfathered_decon_datasets`, so a held dataset is visible in
        # the backfill log rather than merely absent. A recorded fingerprint
        # that is non-NULL and differs is still a real mismatch and still
        # re-submits, so future retunes keep working as designed.
        return True
    return recorded_params == decon_params_fingerprint()


def zarr_deskew_data_dir(ds: LeafDataset, psf: Path | None) -> Path:
    """The ticket `dataDir` for a KIND_ZARR_PRECROPPED dataset.

    Deskew-only reads the cheap symlink mirror; decon reads the materialized
    `(ny, nx, nz)` TIFFs (see `build_decon_staging_dir` for why it cannot
    share the mirror). PetaKit5D writes its DS/DSR/Decon output *inside*
    whichever of these is the `dataDir`, so `_dsr_dir_for` in
    `backfill/cli.py` must resolve through here too.

    Resolved through `resolve_output_base()` like every other output path,
    so a KIND_ZARR_PRECROPPED dataset under an unwritable raw dir gets its
    staging dir mirrored too, rather than failing at mkdir the same way the
    TIFF-path crop stage used to.
    """
    return resolve_output_base(ds.leaf_dir) / ("decon_stage" if psf else "zarr_mirror")


def build_decon_staging_dir(
    channel_zarr_paths: tuple[Path, ...],
    staging_dir: Path,
    *,
    dataset_prefix: str | None = None,
    max_timepoints: int | None = None,
) -> Path:
    """Materialize a zarr-precropped acquisition as per-timepoint TIFFs in the
    `(ny, nx, nz)` layout PetaKit5D's deconvolution requires.

    Why this exists rather than reusing the symlink mirror: deconvolution is
    a 3D convolution, and PetaKit5D's decon path (XR_decon_data_wrapper ->
    XR_RLdeconFrame3D -> RLdecon) has NO axis-order parameter -- it convolves
    the array exactly as stored. `inputAxisOrder='zxy'`, which corrects the
    mirror's `(z, y, x)` layout, is an argument to
    XR_deskew_rotate_data_wrapper and is applied inside XR_deskewRotateFrame,
    i.e. AFTER decon has already run. Deconvolving the mirror directly would
    convolve the 1458-px coverslip axis with the PSF's 81-plane z kernel and
    report success. Permuting the PSF instead does not rescue it: `psf_gen_new`
    resamples PSF dim 3 from dz_psf to dz_data, and
    `omw_backprojector_generation` with `skewed=true` builds its OTF mask as
    `cat(3, ...)` -- both hard-code dim 3 == scan Z.

    So the pixels have to be rewritten. That puts the zarr path onto exactly
    the same footing as the legacy OME-TIFF path, whose cropper has always
    applied the same rot90 (`opym.utils.orient_zyx_for_decon_tiff`) -- which
    is also the orientation the measured PSF itself carries
    (`psf_tools/extract_bead_psf.py` applies `np.rot90(k=1, axes=(1,2))`).
    The ticket is then submitted with `zarr_input=False`, so
    `input_axis_order` derives to `'yxz'` and no permute happens at all.

    Naming matches `build_zarr_pyramid_mirror`'s exactly -- `_C{c}_T{ttt}` for
    a time series, the store's own `.ome`-preserving name for a single
    timepoint -- so `channel_patterns` and `_run_mip_encode`'s output matching
    both work unchanged.

    Symlinking is not an option here (the bytes genuinely differ), so this
    does cost a transposed copy of the raw data; `zlib` keeps it to roughly a
    quarter of the raw size on this dim data. `backfill/cli.py` deletes it
    once `mip_encode` succeeds.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    built: set[Path] = set()
    for cidx, store in enumerate(channel_zarr_paths):
        if not _zarr_store_is_ready(store):
            raise FileNotFoundError(
                f"{store} is missing .zattrs or its main pixel-data array -- "
                "likely still mid-Globus-transfer, will retry on next run"
            )
        pixel_dir = store / _read_ome_zarr_dataset_path(store)
        arr = zarr.open(str(pixel_dir), mode="r")
        # Single-timepoint stores keep the store's own name (minus only
        # `.zarr`, so the `.ome` component survives) because
        # `_run_mip_encode`'s poster branch matches PetaKit5D's MIP output
        # against exactly that -- see its comment.
        single_name = store.name.removesuffix(".zarr") + ".tif"

        if arr.ndim == 3:
            dst = staging_dir / single_name
            built.add(dst)
            write_decon_staged_tiff(arr, dst)
            continue

        # Decide single-vs-multi naming from `channel_store_timepoints`
        # (REAL written chunk dirs) -- the same signal `dataset_timepoints()`
        # uses, and NOT `arr.shape[0]` (the store's raw DECLARED length) or
        # the loop bound `n_t` below (which `max_timepoints` can clamp for
        # an intentionally-fast test run on an otherwise-healthy, genuinely
        # multi-timepoint store -- that must still get multi-style naming,
        # confirmed by test_max_timepoints_caps_the_stage).
        #
        # Confirmed real failure this fixes: an aborted acquisition whose
        # store still DECLARES e.g. shape[0]=2 but only ever wrote 1 real
        # timepoint's chunks (`channel_store_timepoints()` -> 1) took the
        # multi-timepoint branch anyway (old check: `arr.shape[0] == 1`,
        # false here), writing `<prefix>_C0_T000.tif` -- while the caller,
        # going by `dataset_timepoints()`, submitted the ticket with the
        # single-timepoint pattern `<name>.ome`, matching nothing on disk.
        # PetaKit5D's `getImageSize('')` then died with "Index exceeds
        # array bounds".
        if channel_store_timepoints(store) <= 1:
            dst = staging_dir / single_name
            built.add(dst)
            write_decon_staged_tiff(arr[0], dst)
            continue

        n_t = arr.shape[0] if max_timepoints is None else min(arr.shape[0], max_timepoints)
        prefix = dataset_prefix or parse_zarr_group_prefix(store)
        for t in range(n_t):
            if not (pixel_dir / str(t)).is_dir():
                # A store can declare more timepoints than it wrote (aborted
                # acquisition); callers clamp, but don't take the dataset
                # down if one slips through.
                print(f"   Skipping T{t:03d} of {store.name}: no chunk data")
                continue
            dst = staging_dir / f"{prefix}_C{cidx}_T{t:03d}.tif"
            built.add(dst)
            write_decon_staged_tiff(arr[t], dst)

    # Same stale-output hazard the mirror guards against: a staging dir built
    # when more timepoints existed would leave frames that still match
    # `_C{c}_T` and make the per-channel counts diverge downstream.
    for entry in staging_dir.glob("*.tif"):
        if entry not in built and entry.is_file():
            print(f"   Removing stale staged frame {entry.name}")
            entry.unlink()
    for entry in staging_dir.glob("*.tif.tmp"):
        entry.unlink()
    return staging_dir


def _live_lane_owns_deskew(ds: LeafDataset, registry: StatusRegistry, decon_psf: Path | None) -> bool:
    """Hand-off from the live lane (opym.stream.live), which deconvolves and
    deskews a streamed acquisition timepoint by timepoint into this dataset's
    own DSR directory and records the outcome in `.live_status.json` there.

    True (skip submitting) when the live lane is still processing this
    dataset, or has finished it with the same PSF and decon parameters and
    every timepoint's frames are on disk; the latter also records deskew as
    done, so only mip_encode and the viewer export remain. False otherwise
    (no live run, failed, stale, or incomplete): the normal batch path runs,
    and its stale-output cleanup removes whatever the live lane left.
    """
    if decon_psf is None:
        return False
    dsr_dir = dsr_output_dir(zarr_deskew_data_dir(ds, decon_psf), decon_psf)
    status = read_live_status(dsr_dir)
    if not status:
        return False
    if status.get("state") == "running" and live_status_is_fresh(status):
        print(f"[backfill] {ds.dataset_key}: live lane still processing it, deferring")
        return True
    if status.get("state") != "complete":
        return False
    if status.get("decon_psf") != str(decon_psf) or status.get("decon_params") != decon_params_fingerprint():
        return False
    n_t = dataset_timepoints(ds)
    n_c = len(ds.channel_zarr_paths)
    frames = parse_dsr_frame_names(dsr_dir)
    if n_t < 1 or not all((c, t) in frames for c in range(n_c) for t in range(n_t)):
        print(
            f"[backfill] {ds.dataset_key}: live output incomplete "
            f"({len(frames)} of {n_t * n_c} frames), reprocessing in batch"
        )
        return False
    registry.set_decon_psf(ds.dataset_key, str(decon_psf))
    registry.set_decon_params(ds.dataset_key, decon_params_fingerprint())
    registry.finish_stage(ds.dataset_key, "deskew", status="done", output_path=str(dsr_dir))
    print(f"[backfill] {ds.dataset_key}: deskewed live ({n_t} timepoints), skipping batch deskew")
    return True


def parse_dsr_frame_names(dsr_dir: Path) -> set[tuple[int, int]]:
    """(channel, timepoint) of every `<prefix>_C<c>_T<t>.tif` DSR frame."""
    out = set()
    for p in Path(dsr_dir).glob("*_C*_T*.tif"):
        m = re.match(r"^.+_C(\d+)_T(\d+)\.tif$", p.name)
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def submit_zarr_deskew_ticket(ds: LeafDataset, registry: StatusRegistry) -> Path | None:
    """`submit_deskew_ticket`'s counterpart for `KIND_ZARR_PRECROPPED`
    datasets: no crop stage exists for these (already cropped/channel-split
    at capture time), so this dispatches directly against a symlink mirror
    of the raw per-channel zarr stores (see `build_zarr_pyramid_mirror`).

    `channel_patterns` for a single-timepoint dataset are the exact member
    filenames (e.g. `["bead_005_GFP_488.ome.zarr", ...]`), not a bare
    shared prefix -- PetaKit5D's channel matching is substring
    `contains()`-based, and a bare prefix like "cell" would also match an
    unrelated sibling dataset's file. For a time series, the mirror is
    exploded to `<prefix>_C<ch>_T<ttt>.zarr` per timepoint (see
    `build_zarr_pyramid_mirror`), and the patterns become `_C0_T`, `_C1_T`,
    ... -- unambiguous because the mirror directory is this dataset's own
    private output namespace, never shared with a sibling dataset.

    `input_target` is the mirror directory (this dataset's own output
    namespace, not the raw dir) so `submit_remote_deskew_job`'s TIFF-
    redirection logic (which only triggers when `input_target.is_file()`)
    is a no-op here, and PetaKit5D writes its DSR_nodecon output *inside*
    the mirror (`ds.leaf_dir / "zarr_mirror" / "DSR_nodecon"`, confirmed
    against a real completed job -- not a sibling of it, see `_dsr_dir_for`
    in `backfill/cli.py`) -- i.e. still fully within `ds.leaf_dir`, never
    colliding with a sibling dataset sharing the same raw directory.
    """
    decon_psf = resolve_decon_psf()
    if registry.is_stage_done(ds.dataset_key, "deskew") and decon_provenance_matches(
        registry, ds.dataset_key, decon_psf
    ):
        return None

    existing = registry.get_stage(ds.dataset_key, "deskew")
    # Reuse an in-flight ticket only when it is for the configuration we
    # want. Provenance is recorded at SUBMIT time (see set_decon_psf below),
    # so a genuinely running job already matches; a `running` row left over
    # from an interrupted deskew-only pass does not, and would otherwise pin
    # the dataset to a ticket that will never produce decon output.
    if (
        existing
        and existing["status"] == "running"
        and existing["ticket_path"]
        and decon_provenance_matches(registry, ds.dataset_key, decon_psf)
    ):
        return Path(existing["ticket_path"])
    if _live_lane_owns_deskew(ds, registry, decon_psf):
        return None
    if _backfill_lane_closed(ds):
        return None
    data_dir = zarr_deskew_data_dir(ds, decon_psf)
    # Reaching here means neither early return above fired, so we are about to
    # submit -- and therefore what is on disk was produced by a configuration
    # that is NOT the one we are about to run (or by a run that crashed).
    # Clearing it is not just tidy-up, it is required twice over:
    #
    #   * PetaKit5D skips a decon frame whose output already exists
    #     (`if exist(deconFullpath, 'file') ... skip it!`, XR_RLdeconFrame3D.m),
    #     so a Decon/ left behind by a run with a different PSF or wienerAlpha
    #     would be silently reused forever instead of recomputed -- the stage
    #     then reports `done` under the NEW provenance fingerprint while the
    #     pixels are the old ones.
    #   * `Decon/Masks/<fsname>_eroded.zarr` survives the intermediate reaper
    #     (which only removes Decon/*.tif and the psfgen copy), and its mere
    #     existence drives XR_RLdeconFrame3D.m:244 into `rmdirs`, which is not
    #     a function anywhere in the vendored PetaKit5D -- every re-run of an
    #     already-deconvolved dataset died there on 2026-09-21. Shimmed in
    #     opym_local/src/opym/patches/rmdirs.m, but removing Decon/ outright
    #     keeps the pipeline from depending on that shim at all.
    #
    # Gating this on a `failed` row (as it used to be) missed exactly the case
    # that matters: a parameter retune re-submits over a `done` row.
    # Unconditional here, with no "has this pipeline run before" guard: unlike
    # the TIFF path's work_dir, `decon_stage/` is created by this pipeline and
    # nothing else ever writes there, so anything inside it is ours.
    # Trade-off: this deletes good existing output before recomputing, so a
    # re-run that then fails loses the old result. That is the same trade the
    # `failed`-only version already made, now applied consistently.
    _clean_stale_deskew_output(dsr_output_dir(data_dir, decon_psf))
    _clean_stale_deskew_output(data_dir / "Decon")

    mda_settings_file = ds.raw_dir / "MDA_settings.yaml"
    # Prefer the stores' own `z` coordinate array over the MDA_settings.yaml
    # sidecar: the sidecar is not written per-acquisition, so this used to
    # fall through to the 0.3 um default on every single dataset -- while
    # the real step was 0.1 or 0.5 depending on the acquisition. Log the
    # source so a defaulted step is visible in the backfill log instead of
    # silently reaching PetaKit5D as a plausible-looking number.
    #
    # And refuse to guess by default: an interrupted or still-transferring
    # acquisition has no `z` array yet (3 of ~100 datasets on disk), and
    # deskewing one at a made-up 0.3 um just produces another silently
    # wrong-sized volume that reports `done`. Deferring costs nothing -- the
    # stage is retried every backfill pass and self-heals the moment the
    # coordinate arrays land. Set OPYM_ZARR_ALLOW_DEFAULT_Z_STEP=1 to process
    # one anyway, knowingly.
    # OPYM_ZARR_ALLOW_DEFAULT_Z_STEP=1 falls back to 0.3, which is a guess and
    # wrong for every dataset in this corpus (they are 0.1 or 0.5). When a
    # store is missing its coordinate arrays but the real step is known --
    # e.g. from sibling acquisitions in the same session -- name the value
    # with OPYM_ZARR_DEFAULT_Z_STEP=<um> instead of accepting 0.3.
    allow_default = os.environ.get("OPYM_ZARR_ALLOW_DEFAULT_Z_STEP")
    explicit_default = os.environ.get("OPYM_ZARR_DEFAULT_Z_STEP")
    if explicit_default:
        fallback_z: float | None = float(explicit_default)
    elif allow_default:
        fallback_z = 0.3
    else:
        fallback_z = None
    z_step_um, z_step_source = resolve_zarr_z_step(
        ds.channel_zarr_paths,
        mda_settings_file,
        default_z_step=fallback_z,
    )
    if z_step_um is None:
        msg = (
            "cannot determine z step: no 'z' coordinate array in any channel "
            f"store under {ds.raw_dir} and no MDA_settings.yaml. Deferring "
            "rather than deskewing at a guessed scale (set "
            "OPYM_ZARR_ALLOW_DEFAULT_Z_STEP=1 to override)."
        )
        print(f"   [{ds.dataset_key}] SKIP -- {msg}")
        # Distinct from a real failure: this is a deliberate guard, cheap to
        # recheck (no multi-GB read, just zarr coordinate-array metadata),
        # and self-heals the moment the data lands -- see SIGNAL_FLAGS'
        # 'blocked' entry. The stage itself still has to record 'failed'
        # (finish_stage only accepts 'done'/'failed'), so the dashboard
        # separates it out via signal_flag, not stage status. set_signal_flag,
        # not set_triage: this dataset already passed roi_detect (deskew is
        # downstream of it), so it has a real expected/actual_timepoints
        # from that pass -- set_triage would null both out on every retry.
        registry.set_signal_flag(ds.dataset_key, "blocked")
        registry.finish_stage(ds.dataset_key, "deskew", status="failed", error=msg)
        return None
    print(f"   [{ds.dataset_key}] z step {z_step_um} um (from {z_step_source})")

    n_timepoints = dataset_timepoints(ds)
    env_cap = int(os.environ.get("OPYM_ZARR_MAX_TIMEPOINTS", "0")) or None
    # Cap the mirror at the timepoints every channel actually wrote rather
    # than the count the stores declare (see `channel_store_timepoints`).
    max_t = n_timepoints if env_cap is None else min(n_timepoints, env_cap)
    if decon_psf:
        # Decon needs real (ny, nx, nz) pixels, not the mirror's (z, y, x)
        # view -- see build_decon_staging_dir. Submitting with
        # zarr_input=False then makes input_axis_order derive to 'yxz', i.e.
        # no permute, exactly like the legacy OME-TIFF path.
        input_dir = build_decon_staging_dir(
            ds.channel_zarr_paths,
            data_dir,
            dataset_prefix=ds.leaf_dir.name,
            max_timepoints=max_t,
        )
    else:
        input_dir = build_zarr_pyramid_mirror(
            ds.channel_zarr_paths,
            data_dir,
            dataset_prefix=ds.leaf_dir.name,
            max_timepoints=max_t,
        )
    if n_timepoints > 1:
        # Exploded per-timepoint frames -> match every frame of a channel.
        # Safe (not the usual full-filename pattern) because the input dir
        # is this dataset's own private namespace, no sibling collision.
        channel_patterns = [f"_C{i}_T" for i in range(len(ds.channel_zarr_paths))]
    else:
        # Strip only the container extension, so the `.ome` component
        # survives and the pattern matches both the mirror's `<name>.ome.zarr`
        # and the staged `<name>.ome.tif`.
        channel_patterns = [p.name.removesuffix(".zarr") for p in ds.channel_zarr_paths]

    ticket_path = _admitted_submit(
        input_target=input_dir,
        z_step_um=z_step_um,
        deskew=True,
        rotate=True,
        psf_path=decon_psf,
        channel_patterns=channel_patterns,
        save_mip=True,
        **deskew_decon_kwargs(decon_psf),
        zarr_input=decon_psf is None,
    )
    if ticket_path is None:
        return None
    registry.set_decon_psf(ds.dataset_key, str(decon_psf) if decon_psf else None)
    registry.set_decon_params(ds.dataset_key, decon_params_fingerprint() if decon_psf else None)
    registry.start_stage(ds.dataset_key, "deskew", ticket_path=str(ticket_path))
    # The MIPs on disk were made from the output this ticket replaces. A
    # `done` row left behind made opym-dashboard report ~450 re-submitted
    # datasets as finished, showing their old movies, for days.
    registry.reset_stage(ds.dataset_key, "mip_encode")
    return ticket_path


def process_zarr_precropped_dataset(ds: LeafDataset, registry: StatusRegistry) -> Path | None:
    """Phase A worker entry point for `KIND_ZARR_PRECROPPED` datasets --
    `process_crop_and_submit`'s counterpart, minus the crop stage (already
    done at capture time). Registers the dataset and submits the deskew
    ticket directly; returns the ticket path for Phase B to poll, same
    contract as `process_crop_and_submit`.
    """
    registry.register_dataset(
        ds.dataset_key,
        root=str(ds.root),
        leaf_dir=str(ds.leaf_dir),
        master_file=str(ds.master_file),
        has_legacy_decon=False,
    )
    try:
        return submit_zarr_deskew_ticket(ds, registry)
    except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest
        print(f"[backfill] {ds.dataset_key}: {e}")
        return None


def process_crop_and_submit(
    ds: LeafDataset, registry: StatusRegistry, *, has_legacy_decon: bool = False
) -> Path | None:
    """Phase A worker entry point: register the dataset, detect ROIs, crop
    to both output formats, and submit the deskew-only ticket. Returns the
    submitted ticket path (for Phase B to poll), or None if nothing needs
    watching (already fully done, or this dataset failed -- in which case
    the failure is already recorded in the registry, not raised further).
    """
    registry.register_dataset(
        ds.dataset_key,
        root=str(ds.root),
        leaf_dir=str(ds.leaf_dir),
        master_file=str(ds.master_file),
        has_legacy_decon=has_legacy_decon,
    )
    try:
        # Already cropped: go straight to the deskew submit, whose own
        # provenance check decides whether there is anything to do. Neither
        # ROI detection nor the crop depends on decon settings, and
        # `detect_rois` returns early for any dataset whose mip_encode was
        # ever done -- which hid every re-submission (a decon retune, a
        # finished ticket to collect) behind "already finished".
        tiff_out_dir = derive_paths(ds.master_file, OutputFormat.TIFF_SERIES).output_dir
        if registry.is_stage_done(ds.dataset_key, "crop_tiff") and _output_looks_present(
            tiff_out_dir
        ):
            return submit_deskew_ticket(ds, tiff_out_dir, registry)

        top_roi, bot_roi, _signal_flag = detect_rois(ds, registry)
        if top_roi is None and bot_roi is None:
            # `detect_rois` returns this pair only as a "nothing more to do
            # here" sentinel -- either the dataset is already fully done
            # (mip_encode 'unknown' skip), or it's a known dead/corrupt raw
            # file being sticky-skipped rather than re-read. A REAL result
            # (even the 'dud' fallback) always has at least one real ROI.
            # Calling crop_and_convert with (None, None) would reach
            # run_processing_job's own ValueError -- true, but a misleading
            # "crop_zarr failed" registry entry for what is really "roi_detect
            # already explains this; nothing to crop."
            return None
        _zarr_out_dir, tiff_out_dir = crop_and_convert(ds, top_roi, bot_roi, registry)
        return submit_deskew_ticket(ds, tiff_out_dir, registry)
    except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest of the run
        print(f"[backfill] {ds.dataset_key}: {e}")
        return None
