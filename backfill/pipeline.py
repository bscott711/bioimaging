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
from opym.petakit import resolve_deskew_working_dir, submit_remote_deskew_job
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

# Deconvolution settings, fixed here rather than left to PetaKit5D's defaults.
# All four fail quietly if left to PetaKit5D's own defaults:
#   * wienerAlpha defaults to 0.005, visibly over-sharpened on these volumes.
#     0.02 was the decon-order comparison's value; a low-SNR-focused 22-variant
#     then 20-variant refinement sweep on Cell_005 (see README.md's "Decon
#     parameter tuning" section for the two review artifacts) picked 0.20,
#     paired with the hann/damp changes below -- alone, higher alpha only
#     marginally helped.
#   * hannWinBounds defaults to [0.8, 1.0]; lowering the lower bound to 0.4
#     (more apodization) is one of the three knobs the sweep's winning
#     "super4" combination changed together.
#   * dampFactor defaults to 1 (off, decon_lucy_omw_function.m); 2 caps a
#     decon value's departure from its own input by 2x -- the direct remedy
#     for isolated over-sharpened voxel spikes the sweep was counting.
#   * edgeErosion defaults to 0, which leaves a bright ringing stripe along the
#     slab boundary -- RLdecon.m applies `edgetaper` per z-PLANE, so the axial
#     faces are never tapered and the FFT wraps there. Eroding 3 voxels removes
#     it, for ~6% of the imaged slab. Unchanged by the sweep above.
# Changing any of these changes what the output looks like, so they belong
# in the ticket (and therefore the log) rather than in a MATLAB default.
DECON_WIENER_ALPHA = 0.20
DECON_OTF_CUM_THRESH = 0.90  # unchanged from the old default; the sweep's super4 kept it
DECON_HANN_WIN_BOUNDS = [0.4, 1.0]
DECON_DAMP_FACTOR = 2
DECON_EDGE_EROSION = 3



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
    if existing and existing["status"] == "failed":
        try:
            work_dir = resolve_deskew_working_dir(ds.master_file)
            _clean_stale_deskew_output(dsr_output_dir(work_dir, decon_psf))
            # See submit_zarr_deskew_ticket: PetaKit5D reuses an existing
            # Decon frame rather than recomputing it, so a stale one (from a
            # failed run, or a different PSF/alpha) must go.
            _clean_stale_deskew_output(work_dir / "Decon")
        except FileNotFoundError:
            pass

    paths = derive_paths(ds.master_file, OutputFormat.ZARR)  # only used for its metadata_file path
    z_step_um = parse_z_step(paths.metadata_file, default_z_step=0.3)
    channel_patterns_str = scan_channel_patterns(tiff_out_dir)
    channel_patterns = channel_patterns_str.split(", ") if channel_patterns_str else None

    ticket_path = submit_remote_deskew_job(
        input_target=ds.master_file,
        z_step_um=z_step_um,
        deskew=True,
        rotate=True,
        psf_path=decon_psf,
        dsr_dir_name=dsr_dir_name_for(decon_psf),
        channel_patterns=channel_patterns,
        wiener_alpha=DECON_WIENER_ALPHA,
        otf_cum_thresh=DECON_OTF_CUM_THRESH,
        hann_win_bounds=DECON_HANN_WIN_BOUNDS,
        damp_factor=DECON_DAMP_FACTOR,
        edge_erosion=DECON_EDGE_EROSION,
        # Without this the ticket carries gpu_decon:false and PetaKit5D runs
        # the RL iterations on CPU -- both cards sit at 0% while the parfor
        # pool grinds. The volumes are small in skewed space (~29M voxels),
        # so this fits many times over in 97 GB.
        gpu_decon=True,
        save_mip=True,
    )
    registry.set_decon_psf(ds.dataset_key, str(decon_psf) if decon_psf else None)
    registry.set_decon_params(ds.dataset_key, decon_params_fingerprint() if decon_psf else None)
    registry.start_stage(ds.dataset_key, "deskew", ticket_path=str(ticket_path))
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


def resolve_decon_psf() -> Path | None:
    """The PSF deconvolution should run with, or None for deskew-only.

    Read from the `OPYM_DECON_PSF` environment variable rather than threaded
    through as an argument, matching how the other run-scoped switches here
    work (`OPYM_ZARR_MAX_TIMEPOINTS`, `OPYM_ZARR_ALLOW_DEFAULT_Z_STEP`): the
    backfill fans datasets out across a process pool, and an env var is
    inherited by every worker without changing any worker signature.
    `run_backfill_cli.py --decon-psf` sets it.

    Unset means today's behavior exactly -- no decon, `DSR_nodecon`.
    """
    raw = os.environ.get("OPYM_DECON_PSF", "").strip()
    if not raw:
        return None
    psf = Path(raw).expanduser()
    if not psf.is_file():
        raise FileNotFoundError(
            f"OPYM_DECON_PSF points at {psf}, which is not a file. Refusing to "
            "fall back to deskew-only silently -- unset it to run without decon."
        )
    return psf.resolve()


def dsr_dir_name_for(psf: Path | None) -> str:
    """Output directory name for a DSR result, keyed on whether decon ran.

    Deconvolved output goes to a DIFFERENT directory than deskew-only output
    so the two can coexist and be compared, and so enabling decon never
    silently overwrites the existing no-decon archive. Both
    `submit_*_deskew_ticket` (which names the output) and `_dsr_dir_for` in
    `backfill/cli.py` (which finds it again afterwards) must derive it from
    here, or the reader looks in the wrong place -- the exact bug class
    `_dsr_dir_for`'s own docstring documents twice.

    The bare name `DSR` is deliberately not used: the PSF-tuning harnesses
    (`psf_tools/sweep_deskew_angles.py`, `psf_tools/omw_rl_comparison.py`)
    already write unrelated output under that name.
    """
    return "DSR_decon" if psf else "DSR_nodecon"


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


def decon_params_fingerprint() -> str:
    """A short, deterministic fingerprint of the OMW knobs currently in
    effect (the DECON_* constants above). `decon_provenance_matches` compares
    this against what a dataset's on-disk output was actually produced with,
    so a parameter retune -- not just a PSF swap -- is visible too. Confirmed
    real gap: locking in `super4` (same PSF file, new alpha/hann/damp) was
    otherwise indistinguishable from a no-op to every already-deskewed
    dataset, since `decon_provenance_matches` used to compare only the PSF
    path.
    """
    return (
        f"a{DECON_WIENER_ALPHA}_o{DECON_OTF_CUM_THRESH}_"
        f"h{DECON_HANN_WIN_BOUNDS[0]}-{DECON_HANN_WIN_BOUNDS[1]}_d{DECON_DAMP_FACTOR}"
    )


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
    return registry.get_decon_params(dataset_key) == decon_params_fingerprint()


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
    data_dir = zarr_deskew_data_dir(ds, decon_psf)
    if existing and existing["status"] == "failed":
        _clean_stale_deskew_output(dsr_output_dir(data_dir, decon_psf))
        # PetaKit5D skips a decon frame whose output already exists
        # (`if exist(deconFullpath, 'file')`), so a Decon/ left behind by a
        # failed run -- or by a run with a different PSF or wienerAlpha --
        # would be silently reused forever instead of recomputed.
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

    ticket_path = submit_remote_deskew_job(
        input_target=input_dir,
        z_step_um=z_step_um,
        # Passed explicitly rather than left to the signature default so the
        # value this pipeline actually relies on shows up in the ticket and
        # the log. The acquisition writer records no lateral pixel size
        # (NGFF scale is the placeholder [1,1,1,1], frame_meta says
        # pixel_size_um: 0.0), so there is nothing to read it from -- 0.136
        # is the detection path's known value, unchanged from the legacy
        # TIFF acquisitions.
        xy_pixel_size=0.136,
        deskew=True,
        rotate=True,
        psf_path=decon_psf,
        dsr_dir_name=dsr_dir_name_for(decon_psf),
        channel_patterns=channel_patterns,
        wiener_alpha=DECON_WIENER_ALPHA,
        otf_cum_thresh=DECON_OTF_CUM_THRESH,
        hann_win_bounds=DECON_HANN_WIN_BOUNDS,
        damp_factor=DECON_DAMP_FACTOR,
        edge_erosion=DECON_EDGE_EROSION,
        # Without this the ticket carries gpu_decon:false and PetaKit5D runs
        # the RL iterations on CPU -- both cards sit at 0% while the parfor
        # pool grinds. The volumes are small in skewed space (~29M voxels),
        # so this fits many times over in 97 GB.
        gpu_decon=True,
        save_mip=True,
        zarr_input=decon_psf is None,
    )
    registry.set_decon_psf(ds.dataset_key, str(decon_psf) if decon_psf else None)
    registry.set_decon_params(ds.dataset_key, decon_params_fingerprint() if decon_psf else None)
    registry.start_stage(ds.dataset_key, "deskew", ticket_path=str(ticket_path))
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
