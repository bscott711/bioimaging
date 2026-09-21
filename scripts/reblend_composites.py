"""Re-blends already-encoded composite MIP previews (movie or single-frame
poster) from their source per-channel MIP TIFFs, using the current
`_CHANNEL_COLORS` palette in `backfill/mip_movie.py` -- without re-running
crop/deskew/decon or re-encoding the untouched per-channel movies.

Written for the cyan/magenta -> green/magenta palette change: every dataset
that finished `mip_encode` before that change has a composite baked into
its `mip_movies/` dir in the old colors, and re-running the full backfill
just to fix color would re-touch every stage. This instead reads the same
status registry `backfill/cli.py` writes, which already records each
finished dataset's DSR output dir (`deskew` stage's `output_path`, the
source of the MIP TIFFs) and its `mip_movies/` dir (`mip_encode` stage's
`output_path`, where to overwrite).

Usage:
    uv run python scripts/reblend_composites.py [--dry-run] [--dataset-key KEY]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from opym.registry import StatusRegistry

from backfill.mip_movie import (
    build_poster_for_zarr_dataset,
    find_mip_files,
    load_channel_stack,
    normalize_for_video,
    write_composite,
)

# Mirrors backfill/cli.py's DEFAULT_REGISTRY_PATH (same env var) without
# importing that module's full multiprocessing/MATLAB-ticket machinery just
# for one path constant.
DEFAULT_REGISTRY_PATH = Path(
    os.environ.get(
        "OPYM_BACKFILL_REGISTRY_PATH",
        "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3",
    )
)


def _reblend_movie_composite(dsr_dir: Path, composite_path: Path, fps: float) -> None:
    mips_dir = dsr_dir / "MIPs"
    by_channel = find_mip_files(mips_dir)
    if len(by_channel) < 2:
        raise FileNotFoundError(
            f"expected >=2 channels' MIP TIFFs under {mips_dir}, found {len(by_channel)}"
        )
    normalized_stacks = {
        c: normalize_for_video(load_channel_stack(files)) for c, files in by_channel.items()
    }
    write_composite(normalized_stacks, composite_path, fps=fps)


def _reblend_zarr_precropped_poster(dsr_dir: Path, movies_dir: Path) -> None:
    mips_dir = dsr_dir / "MIPs"
    # Single-timepoint zarr-precropped MIPs have no `_C{c}_T{t}` suffix
    # (see build_poster_for_zarr_dataset's docstring) -- recover each
    # channel's fsname by stripping the fixed `_MIP_z.tif` suffix instead of
    # re-deriving it from `LeafDataset.channel_zarr_paths` (would require
    # re-running full discovery just for this).
    fsnames = sorted(p.name.removesuffix("_MIP_z.tif") for p in mips_dir.glob("*_MIP_z.tif"))
    if not fsnames:
        raise FileNotFoundError(f"no *_MIP_z.tif files under {mips_dir}")
    build_poster_for_zarr_dataset(dsr_dir, fsnames, movies_dir)


def reblend_dataset(dsr_dir: Path, movies_dir: Path, fps: float) -> str:
    """Re-blends whichever composite artifact this dataset has. Returns a
    short description for logging.

    `triage_preview.jpg` alone is NOT a reliable signal for the
    zarr-precropped poster case -- confirmed live against a real
    single-channel legacy dataset that had a stale `triage_preview.jpg`
    left over from `backfill/pipeline.py`'s early, uncolored `roi_detect`-
    stage preview (written before crop/deskew even run, and never cleaned
    up once real per-channel movies exist). Any `.webm` in `movies_dir`
    other than the composite is one of those per-channel movies, which only
    `build_mip_movies_for_dataset` writes -- so its presence (regardless of
    whether a composite ended up alongside it) means this is NOT the
    zarr-precropped single-timepoint poster-only path, and any
    `triage_preview.jpg` sitting next to it is that stale early preview,
    not a color-blended composite.
    """
    composite_candidates = list(movies_dir.glob("*_composite.webm"))
    if composite_candidates:
        _reblend_movie_composite(dsr_dir, composite_candidates[0], fps)
        return "movie composite"
    channel_movies = [p for p in movies_dir.glob("*.webm") if not p.name.endswith("_composite.webm")]
    if channel_movies:
        return "skipped -- single-channel dataset, no composite to re-blend"
    if (movies_dir / "triage_preview.jpg").is_file():
        _reblend_zarr_precropped_poster(dsr_dir, movies_dir)
        return "zarr-precropped poster"
    return "skipped -- no composite/poster found"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--dataset-key", help="Re-blend only this dataset (default: every finished dataset)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be re-blended, without writing"
    )
    args = parser.parse_args()

    registry = StatusRegistry(args.registry_path)
    try:
        datasets = registry.all_datasets()
        if args.dataset_key:
            datasets = [d for d in datasets if d["dataset_key"] == args.dataset_key]

        done = skipped = failed = 0
        for row in datasets:
            key = row["dataset_key"]
            mip_stage = registry.get_stage(key, "mip_encode")
            if not mip_stage or mip_stage["status"] != "done" or not mip_stage["output_path"]:
                continue
            deskew_stage = registry.get_stage(key, "deskew")
            if not deskew_stage or deskew_stage["status"] != "done" or not deskew_stage["output_path"]:
                print(f"[reblend] {key}: skipped -- no completed deskew output_path on record")
                skipped += 1
                continue

            movies_dir = Path(mip_stage["output_path"])
            dsr_dir = Path(deskew_stage["output_path"])
            if args.dry_run:
                print(f"[reblend] {key}: would re-blend ({dsr_dir} -> {movies_dir})")
                continue
            try:
                outcome = reblend_dataset(dsr_dir, movies_dir, args.fps)
                print(f"[reblend] {key}: {outcome}")
                if outcome.startswith("skipped"):
                    skipped += 1
                else:
                    done += 1
            except Exception as e:  # noqa: BLE001 - isolate this dataset's failure from the rest
                print(f"[reblend] {key}: FAILED -- {e}")
                failed += 1

        if not args.dry_run:
            print(f"[reblend] done: {done}, skipped: {skipped}, failed: {failed}")
    finally:
        registry.close()


if __name__ == "__main__":
    main()
