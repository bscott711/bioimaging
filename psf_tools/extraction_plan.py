"""Builds the per-dataset channel extraction plan: which raw (dual-camera)
channel index maps to which output channel, for which fluorophore.

Previously duplicated (with a slightly different, unused 3-tuple variant)
between run_pipeline_cli.py and tests/test_extraction_plan.py; this is now
the single canonical implementation both import from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from psf_tools.analyze_channels import analyze_directory, parse_name_for_fluorophores


@lru_cache(maxsize=None)
def _analyze_directory_cached(base_dir: str) -> tuple[dict, ...]:
    """`analyze_directory` scans every immediate sibling sub-experiment
    folder under `base_dir` each call. A bulk backfill processes many leaf
    datasets that share the same parent `base_dir`, so cache per-directory
    to avoid re-scanning the same siblings once per leaf dataset.
    """
    return tuple(analyze_directory(base_dir))


def get_extraction_plan(target_path: Path) -> list[tuple[int, bool, str, int]]:
    """Returns a list of (raw_channel_index, is_top_camera, laser_name,
    output_channel_index) tuples describing how to remap this dataset's raw
    dual-camera channels into per-excitation output channels.

    `target_path` is the dataset's master `.ome.tif` file; the extraction
    plan is looked up by matching `target_path.parent.name` against
    `analyze_directory(target_path.parent.parent)`'s results -- this
    2-levels-up convention matches the typical
    `<experiment>/<sample>/<master>.ome.tif` layout. For atypical nesting
    (e.g. a single-level `<sample>/<master>.ome.tif` dataset with no
    separate experiment folder), `base_dir` ends up one level too high and
    `analyze_directory` scans a wider tree than strictly needed -- still
    correct, just less efficient. Not fixed here; a known, pre-existing
    heuristic already relied on in production.
    """
    base_dir = target_path.parent.parent
    results = _analyze_directory_cached(str(base_dir))
    folder_name = target_path.parent.name

    match = next((r for r in results if r["Folder"] == folder_name), None)
    if not match:
        return []

    detected = match["Detected_Channels"]
    if not detected:
        detected = parse_name_for_fluorophores(base_dir.name)

    raw_configs = match["Raw_Configs"]

    has_488 = any("488" in c for c in detected)
    has_405 = any("405" in c for c in detected)
    has_561 = any("561" in c for c in detected)
    has_640 = any("640" in c for c in detected)

    plan: list[tuple[int, bool, str, int]] = []
    out_c = 0

    if len(raw_configs) <= 1:
        if has_488:
            plan.append((0, True, "488nm", out_c))
            out_c += 1
        if has_405:
            plan.append((0, False, "405nm", out_c))
            out_c += 1
        if has_561:
            plan.append((1, True, "561nm", out_c))
            out_c += 1
        if has_640:
            plan.append((1, False, "640nm", out_c))
            out_c += 1
    else:
        for i, config in enumerate(raw_configs):
            config_lower = config.lower()
            if "488" in config_lower:
                plan.append((i * 2 + 0, True, "488nm", out_c))
                out_c += 1
            if "405" in config_lower:
                plan.append((i * 2 + 0, False, "405nm", out_c))
                out_c += 1
            if "561" in config_lower:
                plan.append((i * 2 + 1, True, "561nm", out_c))
                out_c += 1
            if "640" in config_lower:
                plan.append((i * 2 + 1, False, "640nm", out_c))
                out_c += 1

    return plan
