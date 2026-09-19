#!/usr/bin/env python3
"""Entry point: opym-backfill -- bulk crop/zarr/deskew/MIP backfill.

Deconvolution is opt-in via --decon-psf; without it this is the
deskew-only pipeline it has always been.

Thin orchestrator, matching the run_pipeline_cli.py/run_napari_opym.py
convention: business logic lives in backfill/, this just parses args.
"""

import argparse
import os
from pathlib import Path

from backfill.cli import DEFAULT_REGISTRY_PATH, run_backfill, watch_backfill

DEFAULT_ROOTS = [
    Path("/mmfs1/scratch/jacks.local/microscopy"),
    Path("/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload"),
    Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload"),
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bulk backfill: crop -> channel-remap -> zarr -> [optional decon] "
            "-> deskew/rotate -> MIP, across every raw OPM acquisition found "
            "under the given data roots. Decon is off unless --decon-psf is given."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=DEFAULT_ROOTS,
        help="Data roots to walk independently (no cross-root deduplication -- see the plan doc).",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="SQLite status registry path (also read by the opym-dashboard repo).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Crop-stage worker processes (default: min(8, cpu_count//2)).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run discovery and print what would be processed -- no writes, no submissions.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help=(
            "Run real discovery and register every dataset as pending in the "
            "status registry (so the dashboard shows the real backlog), but "
            "stop before any crop/deskew/MIP work or MATLAB ticket submission."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between deskew-ticket resolution checks once cropping is done.",
    )
    parser.add_argument(
        "--mip-fps",
        type=float,
        default=12.0,
        help="Frame rate for encoded MIP movies.",
    )
    parser.add_argument(
        "--decon-psf",
        type=Path,
        default=None,
        help=(
            "Deconvolve with this PSF before deskewing, writing to DSR_decon/ "
            "instead of DSR_nodecon/ so existing no-decon output is preserved. "
            "Omit for deskew-only (the default, unchanged). The PSF must be a "
            "measured, skewed-space PSF whose 3rd MATLAB dimension is the scan "
            "axis; see backfill/pipeline.py's build_decon_staging_dir."
        ),
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Run forever instead of once, re-scanning the data roots every SECONDS "
            "so new uploads are picked up without a manual re-run. Intended for a "
            "persistent service (see deploy/opym-backfill.service)."
        ),
    )
    args = parser.parse_args()

    # Exported rather than threaded through as an argument: the backfill fans
    # datasets out across a process pool, and an env var reaches every worker
    # without changing any worker signature -- the same mechanism
    # OPYM_ZARR_MAX_TIMEPOINTS / OPYM_ZARR_ALLOW_DEFAULT_Z_STEP already use.
    # backfill.pipeline.resolve_decon_psf() reads it.
    if args.decon_psf is not None:
        psf = args.decon_psf.expanduser().resolve()
        if not psf.is_file():
            parser.error(f"--decon-psf {psf} is not a file")
        os.environ["OPYM_DECON_PSF"] = str(psf)
        print(f"[backfill] Deconvolution ENABLED with PSF {psf}")
        print("[backfill] Output -> DSR_decon/ (DSR_nodecon/ left untouched)")

    if args.watch is not None:
        watch_backfill(
            roots=args.roots,
            registry_path=args.registry_path,
            workers=args.workers,
            dry_run=args.dry_run,
            discover_only=args.discover_only,
            poll_interval_s=args.poll_interval,
            mip_fps=args.mip_fps,
            watch_interval_s=args.watch,
        )
    else:
        run_backfill(
            roots=args.roots,
            registry_path=args.registry_path,
            workers=args.workers,
            dry_run=args.dry_run,
            discover_only=args.discover_only,
            poll_interval_s=args.poll_interval,
            mip_fps=args.mip_fps,
        )


if __name__ == "__main__":
    main()
