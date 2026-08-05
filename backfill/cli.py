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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from opym.discovery import LeafDataset, discover_leaf_datasets
from opym.registry import StatusRegistry
from opym.utils import sanitize_filename

from backfill.mip_movie import build_mip_movies_for_dataset
from backfill.pipeline import process_crop_and_submit

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


def _crop_worker(
    ds: LeafDataset, registry_path: Path, legacy_decon: bool
) -> tuple[str, str | None]:
    """Runs in its own process -- sqlite3 connections aren't picklable, so
    each worker opens its own `StatusRegistry(registry_path)` rather than
    sharing one across the pool (same rule as "no shared file handles
    across process boundaries" elsewhere in this codebase).
    """
    registry = StatusRegistry(registry_path)
    try:
        ticket_path = process_crop_and_submit(ds, registry, has_legacy_decon=legacy_decon)
        return ds.dataset_key, (str(ticket_path) if ticket_path else None)
    finally:
        registry.close()


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

        dsr_dir = ds.leaf_dir / "processed_tiff_series_split" / "DSR_nodecon"
        registry.finish_stage(dataset_key, "deskew", status="done", output_path=str(dsr_dir))

        registry.start_stage(dataset_key, "mip_encode")
        try:
            movies_dir = ds.leaf_dir / "mip_movies"
            sanitized_name = sanitize_filename(ds.master_file.name)
            build_mip_movies_for_dataset(dsr_dir, sanitized_name, movies_dir, fps=mip_fps)
            registry.finish_stage(dataset_key, "mip_encode", status="done", output_path=str(movies_dir))
        except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest
            registry.finish_stage(dataset_key, "mip_encode", status="failed", error=str(e))
            print(f"[backfill] {dataset_key}: mip_encode failed: {e}")


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

    print(f"[backfill] Phase A+B: cropping ({num_workers} workers) and polling deskew/MIP as tickets resolve...")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_crop_worker, ds, registry_path, legacy_flags[ds.dataset_key]): ds
            for ds in datasets
        }
        for future in as_completed(futures):
            dataset_key, ticket_path_str = future.result()
            if ticket_path_str:
                pending[dataset_key] = Path(ticket_path_str)
            _drain_resolved_tickets(pending, dataset_by_key, registry, mip_fps)

    print(f"[backfill] Phase A complete. {len(pending)} dataset(s) still awaiting deskew resolution.")
    while pending:
        _drain_resolved_tickets(pending, dataset_by_key, registry, mip_fps)
        if pending:
            time.sleep(poll_interval_s)

    registry.close()
    print("[backfill] Done.")
