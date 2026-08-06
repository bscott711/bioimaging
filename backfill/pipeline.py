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
import traceback
from pathlib import Path

import tifffile
import zarr
from opym.core import run_processing_job
from opym.discovery import LeafDataset
from opym.metadata import parse_expected_timepoints, parse_z_step, parse_zarr_z_step
from opym.petakit import submit_remote_deskew_job
from opym.registry import StatusRegistry
from opym.roi_detect import EXPECTED_H, EXPECTED_W, auto_detect_rois, compute_reference_projection
from opym.utils import OutputFormat, derive_paths, scan_channel_patterns
from psf_tools.extraction_plan import get_extraction_plan

from backfill.mip_movie import encode_poster_image, normalize_for_video


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
    """
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
    """Submits a decon-skipped deskew+rotate+MIP ticket via the *existing*
    `submit_remote_deskew_job` -- passing `psf_path=None` is the entire
    decon-skip switch (the ticket's `run_decon` server-side default is
    `~isempty(psf_path)`). Routes through the 'deskew' job type, NOT
    `submit_pipeline_job`'s fused GPU 'pipeline' route, which has no
    skip-decon toggle.

    Returns the ticket path if one is pending resolution (freshly submitted,
    or already submitted by a prior run and not yet resolved), or None if
    this stage is already fully done -- the orchestrator's Phase B poller
    uses the returned path to know what to watch; None means "nothing to
    watch, already finished."
    """
    if registry.is_stage_done(ds.dataset_key, "deskew"):
        return None

    existing = registry.get_stage(ds.dataset_key, "deskew")
    if existing and existing["status"] == "running" and existing["ticket_path"]:
        return Path(existing["ticket_path"])

    paths = derive_paths(ds.master_file, OutputFormat.ZARR)  # only used for its metadata_file path
    z_step_um = parse_z_step(paths.metadata_file, default_z_step=0.3)
    channel_patterns_str = scan_channel_patterns(tiff_out_dir)
    channel_patterns = channel_patterns_str.split(", ") if channel_patterns_str else None

    ticket_path = submit_remote_deskew_job(
        input_target=ds.master_file,
        z_step_um=z_step_um,
        deskew=True,
        rotate=True,
        psf_path=None,
        dsr_dir_name="DSR_nodecon",
        channel_patterns=channel_patterns,
        save_mip=True,
    )
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


def build_zarr_pyramid_mirror(channel_zarr_paths: tuple[Path, ...], mirror_dir: Path) -> Path:
    """Bridges two real, independent compatibility gaps confirmed directly
    against PetaKit5D's source and against live failed tickets -- two of its
    own utilities disagree with each other about what a zarr GROUP path
    should contain, and neither has any awareness of the OME-NGFF standard
    (`multiscales`/`p0`) this acquisition software's zarr writer uses:

    1. `ZarrAdapter.m` (used by `blockedImage`, e.g. `XR_parseImageFilenames`)
       only recognizes ITS OWN multi-resolution pyramid naming (`L_1_1_1` for
       the full-resolution level, written by its own
       `XR_multiresZarrGeneration.m`) -- fails with "Unable to open <path>"
       otherwise.
    2. `getImageSize.m` -- called later, on the same path -- takes a
       completely different shortcut: since the path string ends in "zarr",
       it unconditionally tries to `fopen`+`fread` a `.zarray` file directly
       inside the given path (i.e. treats it as a flat array store, no group
       nesting at all), regardless of whether a `.zgroup` is also present.
       Fails with "Invalid file identifier" (silent `fopen` failure) when
       that assumption doesn't hold (confirmed via a real failed ticket).

    Both are satisfied simultaneously with one mirror: `.zgroup` and
    `.zarray` can coexist as plain files/symlinks in the same directory with
    no OS-level conflict, and `ZarrAdapter.openToRead` checks for `.zgroup`
    first, so `.zarray`'s presence doesn't change which branch it takes.
    (The actual pixel-reading path, `readzarr.m`, tries its own compiled
    zarr reader first but falls back to this same group-aware `ZarrAdapter`
    on any exception, so no third convention needs bridging there.)

    A tiny, read-only, symlink-only mirror per dataset -- no pixel data is
    copied, and the real synced acquisition data is never touched (a hard
    constraint for this pipeline) -- every link lives under `mirror_dir`, in
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

    mirror_dir.mkdir(parents=True, exist_ok=True)
    for store in channel_zarr_paths:
        if not _zarr_store_is_ready(store):
            raise FileNotFoundError(
                f"{store} is missing .zattrs or its main pixel-data array -- "
                "likely still mid-Globus-transfer, will retry on next run"
            )
        mirrored_store = mirror_dir / store.name
        mirrored_store.mkdir(exist_ok=True)
        dataset_path = _read_ome_zarr_dataset_path(store)

        _relink(mirrored_store / "L_1_1_1", (store / dataset_path).resolve())
        _relink(mirrored_store / ".zgroup", (store / ".zgroup").resolve())
        _relink(mirrored_store / ".zarray", (store / dataset_path / ".zarray").resolve())
    return mirror_dir


def submit_zarr_deskew_ticket(ds: LeafDataset, registry: StatusRegistry) -> Path | None:
    """`submit_deskew_ticket`'s counterpart for `KIND_ZARR_PRECROPPED`
    datasets: no crop stage exists for these (already cropped/channel-split
    at capture time), so this dispatches directly against a symlink mirror
    of the raw per-channel zarr stores (see `build_zarr_pyramid_mirror`).

    `channel_patterns` are the exact member filenames (e.g.
    `["bead_005_GFP_488.ome.zarr", "bead_005_mScarlet_561.ome.zarr"]`), not
    a bare shared prefix -- PetaKit5D's channel matching is substring
    `contains()`-based, and a bare prefix like "cell" would also match an
    unrelated sibling dataset's file ("cell_001_mScarlet_561.ome.zarr"). A
    full filename can't accidentally match a different file's name that
    way, so this is the only safe pattern given the confirmed real
    file-naming convention (see `opym.discovery.parse_zarr_group_prefix`).

    `input_target` is the mirror directory (this dataset's own output
    namespace, not the raw dir) so `submit_remote_deskew_job`'s TIFF-
    redirection logic (which only triggers when `input_target.is_file()`)
    is a no-op here, and PetaKit5D writes its DSR_nodecon output as a
    sibling of the mirror -- i.e. still fully within `ds.leaf_dir`, never
    colliding with a sibling dataset sharing the same raw directory.
    """
    if registry.is_stage_done(ds.dataset_key, "deskew"):
        return None

    existing = registry.get_stage(ds.dataset_key, "deskew")
    if existing and existing["status"] == "running" and existing["ticket_path"]:
        return Path(existing["ticket_path"])

    mda_settings_file = ds.raw_dir / "MDA_settings.yaml"
    z_step_um = parse_zarr_z_step(mda_settings_file, default_z_step=0.3)
    channel_patterns = [p.name for p in ds.channel_zarr_paths]
    mirror_dir = build_zarr_pyramid_mirror(ds.channel_zarr_paths, ds.leaf_dir / "zarr_mirror")

    ticket_path = submit_remote_deskew_job(
        input_target=mirror_dir,
        z_step_um=z_step_um,
        deskew=True,
        rotate=True,
        psf_path=None,
        dsr_dir_name="DSR_nodecon",
        channel_patterns=channel_patterns,
        save_mip=True,
        zarr_input=True,
    )
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
