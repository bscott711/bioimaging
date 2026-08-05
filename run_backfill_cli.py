#!/usr/bin/env python3
"""Entry point: opym-backfill -- bulk no-decon crop/zarr/deskew/MIP backfill.

Thin orchestrator, matching the run_pipeline_cli.py/run_napari_opym.py
convention: business logic lives in backfill/, this just parses args.
"""

import argparse
from pathlib import Path

from backfill.cli import DEFAULT_REGISTRY_PATH, run_backfill

DEFAULT_ROOTS = [
    Path("/mmfs1/scratch/jacks.local/microscopy"),
    Path("/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload"),
    Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload"),
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bulk backfill: crop -> channel-remap -> zarr -> deskew/rotate "
            "(decon skipped) -> MIP, across every raw OPM acquisition found "
            "under the given data roots."
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
        default=8.0,
        help="Frame rate for encoded MIP movies.",
    )
    args = parser.parse_args()

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
