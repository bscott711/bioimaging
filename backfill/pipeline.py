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
import shutil
import traceback
from pathlib import Path

import numpy as np
import tifffile
import zarr
from opym.core import run_processing_job
from opym.discovery import LeafDataset, parse_zarr_group_prefix
from opym.metadata import parse_expected_timepoints, parse_z_step, resolve_zarr_z_step
from opym.petakit import resolve_deskew_working_dir, submit_remote_deskew_job
from opym.registry import StatusRegistry
from opym.roi_detect import EXPECTED_H, EXPECTED_W, auto_detect_rois, compute_reference_projection
from opym.utils import (
    OutputFormat,
    derive_paths,
    orient_zyx_for_decon_tiff,
    scan_channel_patterns,
)
from psf_tools.extraction_plan import get_extraction_plan

from backfill.mip_movie import encode_poster_image, normalize_for_video

# Deconvolution settings, fixed here rather than left to PetaKit5D's defaults.
# Both defaults are wrong for this data and both fail quietly:
#   * wienerAlpha defaults to 0.005, which is visibly over-sharpened on these
#     volumes. 0.02 is the value the decon-order comparison was run and judged at.
#   * edgeErosion defaults to 0, which leaves a bright ringing stripe along the
#     slab boundary -- RLdecon.m applies `edgetaper` per z-PLANE, so the axial
#     faces are never tapered and the FFT wraps there. Eroding 3 voxels removes
#     it, for ~6% of the imaged slab.
# Changing either of these changes what the output looks like, so they belong
# in the ticket (and therefore the log) rather than in a MATLAB default.
DECON_WIENER_ALPHA = 0.02
DECON_EDGE_EROSION = 3



class DatasetProcessingError(Exception):
    """Raised for a single dataset's stage failure. Always caught by the
    orchestrator -- never allowed to abort the rest of the backfill."""


def _output_looks_present(path: Path) -> bool:
    """Cheap filesystem sanity check backing the registry's "done" status --
    protects against the registry saying done after someone manually deleted
    the output. Deliberately not an exact T*C file-count match (that needs
    re-opening the source file); existence + non-empty is enough to catch
    the common failure mode without adding real cost to every dataset.
    """
    return path.is_dir() and any(path.iterdir())


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
    """
    if registry.is_stage_done(ds.dataset_key, "mip_encode"):
        return None, None, "unknown"

    registry.start_stage(ds.dataset_key, "roi_detect")
    try:
        z = _open_lazy_zarr(ds.master_file)
        master_roi_path = ds.leaf_dir / "master_roi.json"
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

        actual_timepoints = z.shape[0] if z.ndim >= 5 else 1
        metadata_file = derive_paths(ds.master_file, OutputFormat.ZARR).metadata_file
        expected_timepoints = parse_expected_timepoints(metadata_file)
        registry.set_triage(
            ds.dataset_key,
            signal_flag=signal_flag,
            expected_timepoints=expected_timepoints,
            actual_timepoints=actual_timepoints,
        )
        _write_triage_preview(max_proj, ds.leaf_dir / "mip_movies" / "triage_preview.jpg")

        registry.finish_stage(ds.dataset_key, "roi_detect", status="done")
        return top_roi, bot_roi, signal_flag
    except Exception as e:  # noqa: BLE001 - reported into the registry, then re-raised
        classification = _classify_unreadable_raw_file(e)
        if classification is not None:
            registry.set_triage(ds.dataset_key, signal_flag=classification)
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
    """
    extraction_plan = get_extraction_plan(ds.master_file)
    channels_to_output = _channels_to_output(extraction_plan)

    zarr_out_dir = ds.leaf_dir / "processed_ngff"
    tiff_out_dir = ds.leaf_dir / "processed_tiff_series_split"

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
                cli_log_file=ds.leaf_dir / "opm_roi_log.json",
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
                cli_log_file=ds.leaf_dir / "opm_roi_log.json",
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
        edge_erosion=DECON_EDGE_EROSION,
        # Without this the ticket carries gpu_decon:false and PetaKit5D runs
        # the RL iterations on CPU -- both cards sit at 0% while the parfor
        # pool grinds. The volumes are small in skewed space (~29M voxels),
        # so this fits many times over in 97 GB.
        gpu_decon=True,
        save_mip=True,
    )
    registry.set_decon_psf(ds.dataset_key, str(decon_psf) if decon_psf else None)
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

        if shape[0] == 1:
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


def decon_provenance_matches(registry, dataset_key: str, psf: Path | None) -> bool:
    """True when the output already on disk was made with the PSF we are about
    to use.

    `is_stage_done(..., "deskew")` alone is not enough to skip a dataset: a
    stage is only "done" for the PSF it was done WITH. Switching decon on for a
    corpus that was deskewed without it -- or changing PSF -- otherwise looks
    like a no-op, because every dataset reports itself already complete and
    nothing recomputes. That is the same silent-skip failure mode as
    PetaKit5D's own `if exist(deconFullpath,'file')`.
    """
    recorded = registry.get_decon_psf(dataset_key)
    return recorded == (str(psf) if psf else None)


def zarr_deskew_data_dir(ds: LeafDataset, psf: Path | None) -> Path:
    """The ticket `dataDir` for a KIND_ZARR_PRECROPPED dataset.

    Deskew-only reads the cheap symlink mirror; decon reads the materialized
    `(ny, nx, nz)` TIFFs (see `build_decon_staging_dir` for why it cannot
    share the mirror). PetaKit5D writes its DS/DSR/Decon output *inside*
    whichever of these is the `dataDir`, so `_dsr_dir_for` in
    `backfill/cli.py` must resolve through here too.
    """
    return ds.leaf_dir / ("decon_stage" if psf else "zarr_mirror")


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
    def _write_staged(volume_zyx, dst: Path) -> None:
        if dst.exists():
            return
        oriented = orient_zyx_for_decon_tiff(np.asarray(volume_zyx))
        # Write-then-rename: a half-written TIFF is not merely incomplete, it
        # poisons every subsequent retry, because PetaKit5D's `readtiff`
        # raises on it and the skip-if-present check above would keep handing
        # it back. Same reasoning as `_clean_stale_deskew_output`.
        tmp = dst.with_name(dst.name + ".tmp")
        tifffile.imwrite(tmp, oriented, compression="zlib")
        os.replace(tmp, dst)

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
            _write_staged(arr, dst)
            continue
        if arr.shape[0] == 1:
            dst = staging_dir / single_name
            built.add(dst)
            _write_staged(arr[0], dst)
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
            _write_staged(arr[t], dst)

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
        _zarr_out_dir, tiff_out_dir = crop_and_convert(ds, top_roi, bot_roi, registry)
        return submit_deskew_ticket(ds, tiff_out_dir, registry)
    except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest of the run
        print(f"[backfill] {ds.dataset_key}: {e}")
        return None
