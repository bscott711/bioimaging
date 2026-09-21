#!/usr/bin/env python3
"""Metrics + renders for the low-SNR decon sweep on Cell_005 T000.

Reads every variant's DSR output from the out-of-band sweep driven by
`opym/decon_sweep_driver.m` (see the plan at
~/.claude/plans/in-opym-local-we-now-crispy-possum.md), computes the
trustworthy metrics (background, noise sigma, peak-above-background,
isolated-maxima count, slab-edge ratio -- deliberately NOT a sharpness
scalar, see project-decon-reenable memory: three attempts at one all
pointed the wrong way on this project), and renders the PNGs the review
artifact needs.

Usage: uv run python -m psf_tools.decon_sweep_report --sweep-root <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

CHANNELS = ("C0", "C1")
CHANNEL_LABELS = {"C0": "488 memNG", "C1": "561 mScarlet-2xFYVE"}


def find_dsr_tif(sweep_root: Path, stage_dir: Path, variant_name: str, decon: bool, channel: str) -> Path:
    """The driver's DSR output lands in one of two places depending on
    whether the variant ran decon first:
      - decon arm:    <stage_dir>/Decon_<name>/DSR/Cell_005_<ch>_T000.tif
        (XR_decon_data_wrapper's `resultDirName` nests under its INPUT
        dataDir, i.e. stage_dir, not an arbitrary output root -- confirmed
        by running it.)
      - no-decon arm: <stage_dir>/DSR/Cell_005_<ch>_T000.tif
        (XR_deskew_rotate_data_wrapper's dataDir *is* stage_dir for that
        arm, so DSRDirName='DSR' nests directly under it.)
    Both older/alternate layouts are also checked so this stays robust to
    a driver revision.
    """
    candidates = [
        stage_dir / f"Decon_{variant_name}" / "DSR" / f"Cell_005_{channel}_T000.tif",
        sweep_root / f"Decon_{variant_name}" / "DSR" / f"Cell_005_{channel}_T000.tif",
        sweep_root / variant_name / "DSR" / f"Cell_005_{channel}_T000.tif",
        stage_dir / "DSR" / f"Cell_005_{channel}_T000.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No DSR output found for variant={variant_name} channel={channel}; tried {candidates}"
    )


def mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def build_masks(nodecon_vol: np.ndarray) -> dict:
    """Masks computed once from the no-decon arm of THIS channel, so every
    variant of that channel is scored on identical voxels and the masks
    can't be biased by any variant's own decon artifacts.

    Also precomputes the geometry `edge_shell_ratio` needs (a boundary shell
    and an interior core): those depend only on `interior`, which is fixed
    per channel, not per variant. An earlier version left this inside
    `edge_shell_ratio` itself, so it recomputed it on EVERY one of the 44
    variant/channel calls instead of twice total -- turned a few-minute
    report into an hours-long stall. Do it once here.

    The shell/core geometry itself is iterated `binary_erosion`, not
    `distance_transform_edt`: on this array size (~285M voxels),
    `distance_transform_edt` measurably hung for 20+ minutes on a single
    call on this (heavily shared, high-load) node -- confirmed via py-spy,
    stuck inside `numpy.indices`' broadcast-assignment fill, which should
    take a couple seconds even loaded. `binary_erosion` at these same
    iteration counts already ran fast just above, so peeling the boundary
    with repeated small-structuring-element erosions is both cheap and
    proven on this machine. It measures city-block-ish distance rather
    than exact Euclidean, which is a fine substitute for "a few-voxel shell
    near the boundary" -- this is a QC ratio, not a physical distance."""
    slab = nodecon_vol > 0
    interior = ndi.binary_erosion(slab, iterations=4)

    med3 = ndi.median_filter(nodecon_vol, size=3)
    bg_vals = nodecon_vol[interior]
    bg = float(np.median(bg_vals))
    sigma = mad_sigma(bg_vals)

    cell = (med3 > bg + 5 * sigma) & interior
    cell = ndi.binary_closing(cell, iterations=2)
    empty = interior & ~ndi.binary_dilation(cell, iterations=8)

    edge_shell = ndi.binary_erosion(interior, iterations=1) & ~ndi.binary_erosion(interior, iterations=4)
    edge_core = ndi.binary_erosion(interior, iterations=6)

    return {
        "slab": slab,
        "interior": interior,
        "cell": cell,
        "empty": empty,
        "bg": bg,
        "sigma": sigma,
        "edge_shell": edge_shell,
        "edge_core": edge_core,
    }


def edge_shell_ratio(vol: np.ndarray, masks: dict) -> float:
    """Mean intensity in a thin shell near the slab boundary (1-4 erosion
    steps in), divided by the interior mean (6+ steps in). edgeErosion=3
    should hold this near 0 (previously measured 2.16x without erosion,
    0.00x with it -- see project-decon-reenable memory). Shell/core masks
    are precomputed once per channel in `build_masks` -- only the intensity
    lookup varies here."""
    shell = masks["edge_shell"]
    core = masks["edge_core"]
    if shell.sum() < 100 or not core.any():
        return float("nan")
    shell_mean = float(vol[shell].mean())
    interior_mean = float(vol[core].mean())
    if not interior_mean:
        return float("nan")
    return shell_mean / interior_mean


def isolated_maxima_count(vol: np.ndarray, cell: np.ndarray, bg: float, sigma: float, thresh_k: float = 12.0) -> int:
    """3x3x3 local maxima whose value exceeds their 26-neighbour max by a
    margin AND sit above bg + thresh_k*sigma -- the over-sharpening
    signature (isolated 2-3 voxel spikes) counted in earlier sweeps."""
    thresh = bg + thresh_k * sigma
    footprint = np.ones((3, 3, 3), dtype=bool)
    footprint[1, 1, 1] = False
    local_max = ndi.maximum_filter(vol, footprint=footprint)
    spikes = cell & (vol > thresh) & (vol.astype(np.float32) > local_max.astype(np.float32) + sigma)
    return int(spikes.sum())


def compute_variant_metrics(vol: np.ndarray, masks: dict) -> dict:
    empty = masks["empty"]
    cell = masks["cell"]
    bg_vals = vol[empty]
    bg = float(np.median(bg_vals)) if bg_vals.size else float("nan")
    sigma = mad_sigma(bg_vals) if bg_vals.size else float("nan")
    cell_vals = vol[cell]
    peak = float(np.percentile(cell_vals, 99.99)) if cell_vals.size else float("nan")
    peak_minus_bg = peak - bg
    n_spikes = isolated_maxima_count(vol, cell, bg, sigma if sigma > 0 else 1.0)
    frac_clipped = float((cell_vals <= 0).mean()) if cell_vals.size else float("nan")
    edge_ratio = edge_shell_ratio(vol, masks)
    noise_rank = sigma / peak_minus_bg if peak_minus_bg and peak_minus_bg > 0 else float("nan")
    return {
        "bg": bg,
        "sigma": sigma,
        "peak_p9999": peak,
        "peak_minus_bg": peak_minus_bg,
        "isolated_maxima_gt12sigma": n_spikes,
        "frac_cell_clipped_zero": frac_clipped,
        "edge_shell_ratio": edge_ratio,
        "noise_over_peak_RANKING_ONLY": noise_rank,
    }


def run(sweep_root: Path, variants_json: Path, render_dir: Path) -> dict:
    cfg = json.loads(variants_json.read_text())
    stage_dir = Path(cfg["stage_dir"])
    variants = cfg["variants"]
    render_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}
    masks_by_channel: dict[str, dict] = {}
    vols_by_variant_channel: dict[tuple[str, str], np.ndarray] = {}

    # Pass 1: load every volume once (also needed for rendering).
    for v in variants:
        name = v["name"]
        decon = bool(v.get("decon", False))
        for ch in CHANNELS:
            path = find_dsr_tif(sweep_root, stage_dir, name, decon, ch)
            vol = tifffile.imread(path)
            print(f"loaded {name}/{ch}: {path} shape={vol.shape} dtype={vol.dtype} "
                  f"min={vol.min()} max={vol.max()}")
            vols_by_variant_channel[(name, ch)] = vol.astype(np.float32)

    # Pass 2: build masks from nodecon per channel.
    for ch in CHANNELS:
        nodecon_vol = vols_by_variant_channel[("nodecon", ch)]
        masks_by_channel[ch] = build_masks(nodecon_vol)
        m = masks_by_channel[ch]
        print(f"[masks] {ch}: interior={m['interior'].sum()} cell={m['cell'].sum()} "
              f"empty={m['empty'].sum()} nodecon_bg={m['bg']:.2f} sigma={m['sigma']:.2f}")

    # Pass 3: metrics, one pass, intermediates printed.
    for v in variants:
        name = v["name"]
        row = {}
        for ch in CHANNELS:
            vol = vols_by_variant_channel[(name, ch)]
            m = compute_variant_metrics(vol, masks_by_channel[ch])
            print(f"[metrics] {name}/{ch}: " + ", ".join(f"{k}={val:.3g}" if isinstance(val, float) else f"{k}={val}" for k, val in m.items()))
            row[ch] = m
        metrics[name] = row

    metrics_path = render_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {metrics_path}")

    return {
        "metrics": metrics,
        "vols": vols_by_variant_channel,
        "masks": masks_by_channel,
        "variants": variants,
        "stage_dir": stage_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--variants-json", type=Path, default=None)
    ap.add_argument("--render-dir", type=Path, default=None)
    args = ap.parse_args()

    variants_json = args.variants_json or (args.sweep_root / "variants.json")
    render_dir = args.render_dir or (args.sweep_root / "render")
    run(args.sweep_root, variants_json, render_dir)


if __name__ == "__main__":
    main()
