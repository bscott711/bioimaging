#!/usr/bin/env python3
"""Fast-path renderer for the decon sweep artifact: produces every XY/ortho
PNG and the psfgen copies WITHOUT computing masks or metrics, so the
artifact has something to show immediately. `decon_sweep_report.run()`
(masks + the trustworthy metrics: background, spike counts, edge ratio) is
the slow path -- median_filter and the morphology chain on a ~285M-voxel
array turned out to take many minutes on this heavily-shared node -- and
runs separately afterward to fill in metrics.json without re-rendering
anything.

Zoom centers here are picked WITHOUT the cell/empty masks (a plain argmax
and a fixed background-ish corner) since those masks are exactly the slow
part. Good enough for a first look; `decon_sweep_report`'s masks are more
principled and metrics.json carries the numbers that matter.

Usage: uv run python -m psf_tools.decon_sweep_quickrender --sweep-root <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from psf_tools.decon_sweep_report import CHANNEL_LABELS, CHANNELS, find_dsr_tif

ZOOM_SIZE = 256


def render_xy(vol, vmin, vmax, out_png: Path, title: str):
    mip = vol.max(axis=0)
    fig, ax = plt.subplots(figsize=(6, 6 * mip.shape[0] / mip.shape[1]))
    ax.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def render_ortho(vol, zoom, vmin, vmax, out_png: Path, title: str):
    z, (py0, px0), (ey0, ex0) = zoom["z"], zoom["puncta"], zoom["empty"]
    half = ZOOM_SIZE
    xz = vol.max(axis=1)
    yz = vol.max(axis=2)
    puncta_zoom = vol[z, py0:py0 + half, px0:px0 + half]
    empty_zoom = vol[z, ey0:ey0 + half, ex0:ex0 + half]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(xz, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("XZ MIP", fontsize=9)
    axes[1].imshow(yz, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("YZ MIP", fontsize=9)
    axes[2].imshow(puncta_zoom, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"puncta zoom z={z}", fontsize=9)
    axes[3].imshow(empty_zoom, cmap="gray", vmin=vmin, vmax=vmax)
    axes[3].set_title(f"empty-slab zoom z={z}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def pick_zoom_naive(nodecon_c1: np.ndarray) -> dict:
    """No masks: brightest voxel in the whole volume for the puncta zoom
    (this data's real signal easily dominates background, confirmed by the
    photon-budget numbers -- background is ~100 counts, decon peaks are
    already measured in the hundreds to ~1800), and a fixed offset corner
    for the empty-slab zoom (the slab is much larger than the zoom box, so
    a geometric corner well inside the eroded interior is background with
    near-certainty for this geometry)."""
    nz, ny, nx = nodecon_c1.shape
    idx = np.unravel_index(np.argmax(nodecon_c1), nodecon_c1.shape)
    z, cy, cx = idx
    half = ZOOM_SIZE // 2

    def clamp_box(cy, cx):
        y0 = int(np.clip(cy - half, 0, max(ny - ZOOM_SIZE, 0)))
        x0 = int(np.clip(cx - half, 0, max(nx - ZOOM_SIZE, 0)))
        return y0, x0

    py0, px0 = clamp_box(cy, cx)
    # A point well inside the slab, off to one side of the puncta -- offset
    # by a large fraction of the frame from the bright spot so it's very
    # unlikely to double up on real structure.
    ey0, ex0 = clamp_box(ny // 5, nx // 2)
    return {"z": int(z), "puncta": (py0, px0), "empty": (ey0, ex0)}


def find_psfgen_figure(sweep_root: Path, stage_dir: Path, variant_name: str):
    for decon_dir in (stage_dir / f"Decon_{variant_name}", sweep_root / f"Decon_{variant_name}"):
        hits = list(decon_dir.glob("psfgen/*_figure.png"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, required=True)
    args = ap.parse_args()

    sweep_root = args.sweep_root
    render_dir = sweep_root / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((sweep_root / "variants.json").read_text())
    stage_dir = Path(cfg["stage_dir"])
    variants = cfg["variants"]

    vols = {}
    for v in variants:
        name = v["name"]
        decon = bool(v.get("decon", False))
        for ch in CHANNELS:
            path = find_dsr_tif(sweep_root, stage_dir, name, decon, ch)
            vol = tifffile.imread(path).astype(np.float32)
            vols[(name, ch)] = vol
            print(f"loaded {name}/{ch}")

    display_range = {}
    zoom_by_channel = {}
    for ch in CHANNELS:
        prod_vol = vols[("prod", ch)]
        nonzero = prod_vol[prod_vol > 0]
        vmin = float(np.percentile(nonzero, 0.5)) if nonzero.size else 0.0
        vmax = float(np.percentile(nonzero, 99.95)) if nonzero.size else 1.0
        display_range[ch] = (vmin, vmax)
        zoom_by_channel[ch] = pick_zoom_naive(vols[("nodecon", ch)])
        print(f"[display] {ch}: vmin={vmin:.1f} vmax={vmax:.1f} zoom={zoom_by_channel[ch]}")

    psfgen_dir = render_dir / "psfgen"
    psfgen_dir.mkdir(exist_ok=True)
    psfgen_by_variant = {}
    for v in variants:
        name = v["name"]
        fig = find_psfgen_figure(sweep_root, stage_dir, name)
        if fig is not None:
            dst = psfgen_dir / f"{name}_psfgen_figure.png"
            shutil.copy(fig, dst)
            psfgen_by_variant[name] = f"psfgen/{dst.name}"

    manifest = {"channels": {}, "variants": [v["name"] for v in variants], "display_range": display_range,
                "metrics_pending": True}
    for ch in CHANNELS:
        vmin, vmax = display_range[ch]
        zoom = zoom_by_channel[ch]
        manifest["channels"][ch] = {"label": CHANNEL_LABELS[ch], "renders": {}}
        for v in variants:
            name = v["name"]
            vol = vols[(name, ch)]
            xy_png = render_dir / f"xy_{name}_{ch}.png"
            ortho_png = render_dir / f"ortho_{name}_{ch}.png"
            render_xy(vol, vmin, vmax, xy_png, f"{name} / {ch}")
            render_ortho(vol, zoom, vmin, vmax, ortho_png, f"{name} / {ch}")
            entry = {"xy": xy_png.name, "ortho": ortho_png.name}
            if name in psfgen_by_variant:
                entry["psfgen"] = psfgen_by_variant[name]
            manifest["channels"][ch]["renders"][name] = entry
            print(f"rendered {name}/{ch}")

    (render_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {render_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
