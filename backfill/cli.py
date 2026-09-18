"""Bulk orchestrator: discovers every raw OPM acquisition across the given
data roots and runs each through crop -> zarr -> deskew (decon skipped) ->
MIP, skipping whatever the status registry says is already done.

Two-phase, interleaved rather than sequential-then-sequential: deskew ticket
dispatch is asynchronous against the MATLAB watchdog's own queue, so a crop
worker must never block waiting on GPU queue depth -- Phase A's
ProcessPoolExecutor keeps cropping the next dataset while Phase B's polling
loop opportunistically checks every currently-pending ticket each time a new
one is submitted, so MIP encoding for early-finished datasets starts well
before the last dataset has even been cropped.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

from opym.discovery import KIND_ZARR_PRECROPPED, LeafDataset, discover_leaf_datasets
from opym.petakit import resolve_deskew_working_dir
from opym.registry import StatusRegistry
from opym.utils import sanitize_filename

from backfill.mip_movie import (
    build_mip_movies_for_dataset,
    build_poster_for_zarr_dataset,
    find_mip_files,
)
from backfill.pipeline import (
    dataset_timepoints,
    detect_rois,
    process_crop_and_submit,
    process_zarr_precropped_dataset,
)

# Crop-queue submission order, lowest first -- 'ok' (real signal detected)
# and 'unknown' (triage itself failed, or this is a KIND_ZARR_PRECROPPED
# dataset that skips triage entirely -- see run_backfill's Phase 0; don't
# penalize either case) go first, 'dud' (no detectable signal in either
# camera half) goes last so GPU/CPU time is spent on likely-good data
# before likely-empty data.
_SIGNAL_PRIORITY = {"ok": 0, "unknown": 1, "dud": 2}

DEFAULT_REGISTRY_PATH = Path(
    os.environ.get(
        "OPYM_BACKFILL_REGISTRY_PATH",
        "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3",
    )
)

_LEGACY_DECON_NAMES = ("Decon", "decon")


def has_legacy_decon(leaf_dir: Path) -> bool:
    """Informational only -- every new output path this pipeline writes is
    a distinctly-named additive sibling of `Decon/`/`decon/`, so datasets
    that already have legacy decon output flow through the exact same
    per-stage pipeline as everything else; this just flags them for the
    dashboard.
    """
    return any((leaf_dir / name).is_dir() for name in _LEGACY_DECON_NAMES)


def _ticket_resolved(ticket_path: Path) -> tuple[bool, bool]:
    """(is_done, is_failed) -- mirrors `opym.petakit.wait_for_job`'s
    completed/failed marker convention (same filename, moved from
    `queue/` to `completed/` or `failed/`), checked once rather than
    blocking in a sleep loop.
    """
    base_dir = ticket_path.parent.parent
    done = (base_dir / "completed" / ticket_path.name).exists()
    failed = (base_dir / "failed" / ticket_path.name).exists()
    return done, failed


def _triage_worker(ds: LeafDataset, registry_path: Path) -> tuple[str, str]:
    """Runs in its own process (same reason as `_crop_worker`): a cheap
    upfront signal-presence + frame-count check (one lazy plane read, no
    crop) used purely to order the crop queue -- see `run_backfill`'s Phase
    0. Never lets a triage failure abort the batch; an unclassifiable
    dataset is just deprioritized as 'unknown' rather than 'dud' (so a
    transient read error doesn't get treated the same as confirmed-empty
    data), and still gets a real triage attempt again inside Phase A's
    `process_crop_and_submit` (`detect_rois` is cheap and idempotent).
    """
    registry = StatusRegistry(registry_path)
    try:
        try:
            _top, _bot, signal_flag = detect_rois(ds, registry)
        except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest
            print(f"[backfill] {ds.dataset_key}: triage failed: {e}")
            signal_flag = "unknown"
        return ds.dataset_key, signal_flag
    finally:
        registry.close()


def _crop_worker(
    ds: LeafDataset, registry_path: Path, legacy_decon: bool
) -> tuple[str, str | None]:
    """Runs in its own process -- sqlite3 connections aren't picklable, so
    each worker opens its own `StatusRegistry(registry_path)` rather than
    sharing one across the pool (same rule as "no shared file handles
    across process boundaries" elsewhere in this codebase).

    Dispatches by `ds.kind`: `KIND_ZARR_PRECROPPED` datasets are already
    cropped/channel-split at capture time, so they skip straight to ticket
    submission with no crop stage at all.
    """
    registry = StatusRegistry(registry_path)
    try:
        if ds.kind == KIND_ZARR_PRECROPPED:
            ticket_path = process_zarr_precropped_dataset(ds, registry)
        else:
            ticket_path = process_crop_and_submit(ds, registry, has_legacy_decon=legacy_decon)
        return ds.dataset_key, (str(ticket_path) if ticket_path else None)
    finally:
        registry.close()


def _dsr_dir_for(ds: LeafDataset) -> Path:
    """Where PetaKit5D actually wrote (or will write) `DSR_nodecon/` for
    this dataset. KIND_ZARR_PRECROPPED has no crop stage, so the ticket's
    `dataDir` is `ds.leaf_dir / "zarr_mirror"` (see
    `submit_zarr_deskew_ticket`'s `mirror_dir` / `build_zarr_pyramid_mirror`)
    -- confirmed against a real completed job that PetaKit5D writes
    `DSR_nodecon` *inside* that `dataDir`, not as its sibling: the previous
    `ds.leaf_dir / "DSR_nodecon"` (one level too shallow) caused every real
    zarr-precropped dataset to report "No MIP TIFFs found" even though
    PetaKit5D had already written complete output one directory deeper, at
    `ds.leaf_dir / "zarr_mirror" / "DSR_nodecon"`. Everything else must
    resolve the same way `submit_remote_deskew_job` resolved its `dataDir`
    when the ticket was submitted (prefers the master-stem-named crop dir
    over the legacy `processed_tiff_series_split/` one) -- hardcoding the
    legacy path here caused every dataset that actually used the newer
    convention to report the same "No MIP TIFFs found" symptom even though
    PetaKit5D had already written complete output to the other directory.
    """
    if ds.kind == KIND_ZARR_PRECROPPED:
        return ds.leaf_dir / "zarr_mirror" / "DSR_nodecon"
    return resolve_deskew_working_dir(ds.master_file) / "DSR_nodecon"


def _run_mip_encode(ds: LeafDataset, registry: StatusRegistry, mip_fps: float) -> None:
    dsr_dir = _dsr_dir_for(ds)
    registry.start_stage(ds.dataset_key, "mip_encode")
    try:
        movies_dir = ds.leaf_dir / "mip_movies"
        if ds.kind == KIND_ZARR_PRECROPPED and dataset_timepoints(ds) > 1:
            # Time series: the mirror was exploded to per-timepoint frames
            # named `<prefix>_C<c>_T<ttt>.zarr` (see build_zarr_pyramid_mirror),
            # so PetaKit5D's MIP output is a `_C{c}_T{t}_MIP_z.tif` series --
            # exactly what the legacy movie builder consumes.
            build_mip_movies_for_dataset(dsr_dir, ds.leaf_dir.name, movies_dir, fps=mip_fps)
            by_channel = find_mip_files(dsr_dir / "MIPs")
            actual_t = max((len(v) for v in by_channel.values()), default=0)
            registry.set_triage(
                ds.dataset_key,
                signal_flag="ok",
                expected_timepoints=dataset_timepoints(ds),
                actual_timepoints=actual_t,
            )
        elif ds.kind == KIND_ZARR_PRECROPPED:
            # Single timepoint -- a static poster, not a movie (see
            # build_poster_for_zarr_dataset's docstring). PetaKit5D's real
            # MIP output keeps the ".ome" component (confirmed against a
            # real completed job: "cell_003_GFP_488.ome_MIP_z.tif", not
            # "cell_003_GFP_488_MIP_z.tif") -- strip only ".zarr", not
            # ".ome.zarr", or every real dataset fails to match its own
            # MIP file.
            channel_fsnames = [p.name.removesuffix(".zarr") for p in ds.channel_zarr_paths]
            build_poster_for_zarr_dataset(dsr_dir, channel_fsnames, movies_dir)
        else:
            sanitized_name = sanitize_filename(ds.master_file.name)
            build_mip_movies_for_dataset(dsr_dir, sanitized_name, movies_dir, fps=mip_fps)
        registry.finish_stage(ds.dataset_key, "mip_encode", status="done", output_path=str(movies_dir))
    except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest
        registry.finish_stage(ds.dataset_key, "mip_encode", status="failed", error=str(e))
        print(f"[backfill] {ds.dataset_key}: mip_encode failed: {e}")


def _drain_resolved_tickets(
    pending: dict[str, Path],
    dataset_by_key: dict[str, LeafDataset],
    registry: StatusRegistry,
    mip_fps: float,
) -> None:
    """One non-blocking pass over every currently-pending ticket: resolves
    (marks deskew done/failed +, on success, runs MIP encode) whatever has
    finished, mutating `pending` in place.
    """
    for dataset_key in list(pending.keys()):
        ticket_path = pending[dataset_key]
        done, failed = _ticket_resolved(ticket_path)
        if not (done or failed):
            continue
        del pending[dataset_key]
        ds = dataset_by_key[dataset_key]

        if failed:
            registry.finish_stage(
                dataset_key, "deskew", status="failed",
                error=f"MATLAB job failed -- see {ticket_path.parent.parent / 'failed' / ticket_path.name}",
            )
            continue

        registry.finish_stage(dataset_key, "deskew", status="done", output_path=str(_dsr_dir_for(ds)))
        _run_mip_encode(ds, registry, mip_fps)


def _as_completed_with_timeout(
    pool: ProcessPoolExecutor,
    futures: dict[Future, LeafDataset],
    per_task_timeout_s: float,
) -> Iterator[tuple[Future, LeafDataset]]:
    """Like `concurrent.futures.as_completed(futures)`, except any future
    still running `per_task_timeout_s` after submission is yielded anyway
    (caller must check `future.done()`) instead of being waited on forever.

    Confirmed live that a single pathological dataset (a full-stack
    max-projection over a large single-timepoint calibration stack on a
    slow NFS mount) can occupy a worker for 15+ minutes without ever
    raising -- and `as_completed()` with no timeout, inside a `with
    ProcessPoolExecutor(...):` block, blocks the *entire* pass on it,
    starving every other dataset. Once every future is either done or
    timed out, shuts the pool down without waiting for any
    still-running (abandoned) worker -- it keeps running in the
    background until it finishes on its own, but no longer blocks the
    orchestrator.
    """
    deadlines = {f: time.monotonic() + per_task_timeout_s for f in futures}
    pending = set(futures)
    try:
        while pending:
            next_deadline = min(deadlines[f] for f in pending)
            done, pending = wait(
                pending, timeout=max(0.0, next_deadline - time.monotonic()), return_when=FIRST_COMPLETED
            )
            for f in done:
                yield f, futures[f]
            now = time.monotonic()
            timed_out = {f for f in pending if deadlines[f] <= now}
            pending -= timed_out
            for f in timed_out:
                yield f, futures[f]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# How long a single dataset may occupy a worker before the orchestrator
# stops waiting on it and moves on -- see `_as_completed_with_timeout`.
# Triage (Phase 0) is meant to be a cheap signal-presence check, so a much
# tighter bound than the real crop/convert work (Phase A) catches a
# pathological dataset fast.
_TRIAGE_TASK_TIMEOUT_S = 300.0
_CROP_TASK_TIMEOUT_S = 1800.0


def run_backfill(
    roots: list[Path],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    workers: int | None = None,
    dry_run: bool = False,
    discover_only: bool = False,
    poll_interval_s: float = 30.0,
    mip_fps: float = 12.0,
) -> None:
    """`dry_run`: discovery + print only, zero registry writes -- a fully
    read-only preview. `discover_only`: real discovery + registers every
    dataset as pending in the registry (so the dashboard shows the real
    backlog) but stops before Phase A -- no crop/deskew/MIP work is run, no
    MATLAB tickets are submitted. Neither flag: the full real run.
    """
    datasets = discover_leaf_datasets(roots)
    print(f"[backfill] discovered {len(datasets)} leaf dataset(s) across {len(roots)} root(s)")

    legacy_flags = {ds.dataset_key: has_legacy_decon(ds.leaf_dir) for ds in datasets}
    legacy_count = sum(legacy_flags.values())
    print(
        f"[backfill] {legacy_count} dataset(s) already have legacy Decon/ output "
        "(processed anyway -- new output is an additive sibling folder)"
    )

    if dry_run:
        for ds in datasets:
            flag = " [legacy-decon]" if legacy_flags[ds.dataset_key] else ""
            print(f"  {ds.dataset_key}{flag}")
        return

    registry = StatusRegistry(registry_path)
    # Register every discovered dataset up front so the dashboard shows
    # "pending" work immediately, not only datasets whose first stage has
    # already started.
    for ds in datasets:
        registry.register_dataset(
            ds.dataset_key,
            root=str(ds.root),
            leaf_dir=str(ds.leaf_dir),
            master_file=str(ds.master_file),
            has_legacy_decon=legacy_flags[ds.dataset_key],
        )

    if discover_only:
        registry.close()
        print(f"[backfill] Registered {len(datasets)} dataset(s) as pending. No processing run (--discover-only).")
        return

    num_workers = workers or max(1, min(8, mp.cpu_count() // 2))
    dataset_by_key = {ds.dataset_key: ds for ds in datasets}
    pending: dict[str, Path] = {}

    # KIND_ZARR_PRECROPPED datasets skip triage entirely: detect_rois
    # assumes a raw OME-TIF's dual-camera frame shape, which doesn't apply
    # to already-cropped, already-channel-split zarr input. They default to
    # 'unknown' priority (see _SIGNAL_PRIORITY) -- same as a failed triage,
    # not penalized like a confirmed dud.
    #
    # Datasets already fully done (mip_encode) are also skipped here --
    # detect_rois() itself early-returns for them too (belt-and-suspenders
    # for Phase A's redundant internal call), but filtering them out of the
    # candidate list up front avoids spinning up a worker at all for the
    # common case (most of the registry, on a steady-state watch pass).
    triage_candidates = [
        ds
        for ds in datasets
        if ds.kind != KIND_ZARR_PRECROPPED and not registry.is_stage_done(ds.dataset_key, "mip_encode")
    ]
    print(
        f"[backfill] Phase 0: triaging {len(triage_candidates)} dataset(s) "
        "(signal check + frame count + preview)..."
    )
    signal_flags: dict[str, str] = {}
    pool = ProcessPoolExecutor(max_workers=num_workers)
    futures = {pool.submit(_triage_worker, ds, registry_path): ds for ds in triage_candidates}
    for future, ds in _as_completed_with_timeout(pool, futures, _TRIAGE_TASK_TIMEOUT_S):
        if not future.done():
            print(
                f"[backfill] {ds.dataset_key}: triage timed out after "
                f"{_TRIAGE_TASK_TIMEOUT_S:.0f}s, deprioritizing and moving on "
                "(worker abandoned, may still be running in the background)"
            )
            registry.finish_stage(
                ds.dataset_key, "roi_detect", status="failed",
                error=f"triage timed out after {_TRIAGE_TASK_TIMEOUT_S:.0f}s",
            )
            signal_flags[ds.dataset_key] = "unknown"
            continue
        try:
            dataset_key, flag = future.result()
        except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest
            print(f"[backfill] {ds.dataset_key}: triage worker crashed: {e!r}")
            signal_flags[ds.dataset_key] = "unknown"
        else:
            signal_flags[dataset_key] = flag

    dud_count = sum(1 for f in signal_flags.values() if f == "dud")
    print(
        f"[backfill] Triage complete. {dud_count} likely-dud dataset(s) "
        "deprioritized to the end of the crop queue (still processed, just last)."
    )
    datasets.sort(key=lambda ds: _SIGNAL_PRIORITY.get(signal_flags.get(ds.dataset_key, "unknown"), 1))

    print(f"[backfill] Phase A+B: cropping ({num_workers} workers) and polling deskew/MIP as tickets resolve...")
    pool = ProcessPoolExecutor(max_workers=num_workers)
    futures = {
        pool.submit(_crop_worker, ds, registry_path, legacy_flags[ds.dataset_key]): ds
        for ds in datasets
    }
    for future, ds in _as_completed_with_timeout(pool, futures, _CROP_TASK_TIMEOUT_S):
        if not future.done():
            print(
                f"[backfill] {ds.dataset_key}: crop/deskew-submit timed out after "
                f"{_CROP_TASK_TIMEOUT_S:.0f}s, will retry next pass "
                "(worker abandoned, may still be running in the background)"
            )
            _drain_resolved_tickets(pending, dataset_by_key, registry, mip_fps)
            continue
        try:
            dataset_key, ticket_path_str = future.result()
        except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest
            print(f"[backfill] {ds.dataset_key}: crop worker crashed: {e!r}")
            _drain_resolved_tickets(pending, dataset_by_key, registry, mip_fps)
            continue
        if ticket_path_str:
            pending[dataset_key] = Path(ticket_path_str)
        elif registry.is_stage_done(dataset_key, "deskew") and not registry.is_stage_done(
            dataset_key, "mip_encode"
        ):
            # submit_deskew_ticket/submit_zarr_deskew_ticket return None
            # with no ticket to poll whenever deskew is already 'done' --
            # that's the common "already fully done" case, but it's also
            # exactly what happens when only mip_encode failed on a prior
            # run (e.g. the DSR_nodecon lookup bug _dsr_dir_for fixes):
            # deskew stays 'done' forever and nothing else ever retries
            # mip_encode alone. Catch that case here instead of silently
            # leaving it failed on every subsequent run.
            _run_mip_encode(dataset_by_key[dataset_key], registry, mip_fps)
        _drain_resolved_tickets(pending, dataset_by_key, registry, mip_fps)

    print(f"[backfill] Phase A complete. {len(pending)} dataset(s) still awaiting deskew resolution.")
    while pending:
        _drain_resolved_tickets(pending, dataset_by_key, registry, mip_fps)
        if pending:
            time.sleep(poll_interval_s)

    registry.close()
    print("[backfill] Done.")


def watch_backfill(
    roots: list[Path],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    workers: int | None = None,
    dry_run: bool = False,
    discover_only: bool = False,
    poll_interval_s: float = 30.0,
    mip_fps: float = 12.0,
    watch_interval_s: float = 120.0,
) -> None:
    """Runs `run_backfill` forever, re-discovering and re-processing every
    `watch_interval_s` seconds -- this is what makes new uploads under the
    data roots show up in the dashboard without a human remembering to
    invoke the one-shot CLI. Safe to loop tightly: every stage is
    registry-gated (see `process_crop_and_submit`/`_ticket_resolved`/etc.),
    so a dataset that's already `done` costs one registry lookup per pass,
    not reprocessing.

    One pass's exception doesn't end the service -- caught, logged, and the
    loop sleeps and retries, same resilience philosophy as
    `opym.local_gpu_worker`'s watchdog (a crashed pass shouldn't take the
    whole unattended service down with it).
    """
    print(f"[backfill] Watch mode: re-scanning every {watch_interval_s:.0f}s. Ctrl-C to stop.")
    while True:
        start = time.monotonic()
        try:
            run_backfill(
                roots,
                registry_path=registry_path,
                workers=workers,
                dry_run=dry_run,
                discover_only=discover_only,
                poll_interval_s=poll_interval_s,
                mip_fps=mip_fps,
            )
        except Exception as e:  # noqa: BLE001 - one bad pass must not kill the service
            print(f"[backfill] watch pass failed: {e!r}")
        elapsed = time.monotonic() - start
        print(f"[backfill] Pass took {elapsed:.1f}s. Sleeping {watch_interval_s:.0f}s until next scan.")
        time.sleep(watch_interval_s)
