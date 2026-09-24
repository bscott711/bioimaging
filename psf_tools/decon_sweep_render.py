#!/usr/bin/env python3
"""Renders the PNGs for the low-SNR decon sweep review artifact.

Reuses `decon_sweep_report.run()` for loading + metrics, then produces,
per variant per channel, at a SHARED display range so variants are
visually comparable:
  xy_<variant>_<ch>.png     -- full-frame XY MIP (the wipe-slider layer)
  ortho_<variant>_<ch>.png  -- XZ+YZ MIPs, plus a puncta zoom and an
                               empty-slab zoom picked once from nodecon/C1

Also copies each variant's psfgen QC figure (OTF mask / back projector)
into render/psfgen/ when present.

Usage: uv run python -m psf_tools.decon_sweep_render --sweep-root <dir>
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

from psf_tools.decon_sweep_report import CHANNELS, CHANNEL_LABELS, run

ZOOM_SIZE = 256


def pick_zoom_centers(nodecon_c1: np.ndarray, cell_mask: np.ndarray, empty_mask: np.ndarray) -> dict:
    """One puncta-rich zoom (brightest cell voxel's XY location) and one
    empty-slab zoom (a point deep in the empty mask), both in lab (y,x) at
    a representative z, chosen once so every variant/arm gets the exact
    same crop."""
    nz, ny, nx = nodecon_c1.shape
    z_mid = nz // 2

    cell_z = cell_mask[z_mid]
    if cell_z.any():
        cy, cx = np.unravel_index(np.argmax(np.where(cell_z, nodecon_c1[z_mid], -1)), cell_z.shape)
    else:
        # fall back to global argmax within cell mask
        idx = np.unravel_index(np.argmax(np.where(cell_mask, nodecon_c1, -1)), nodecon_c1.shape)
        z_mid, cy, cx = idx

    empty_z = empty_mask[z_mid]
    if empty_z.any():
        ys, xs = np.where(empty_z)
        mid = len(ys) // 2
        ey, ex = ys[mid], xs[mid]
    else:
        ey, ex = ny // 4, nx // 4

    def clamp_box(cy, cx):
        half = ZOOM_SIZE // 2
        y0 = int(np.clip(cy - half, 0, max(ny - ZOOM_SIZE, 0)))
        x0 = int(np.clip(cx - half, 0, max(nx - ZOOM_SIZE, 0)))
        return y0, x0

    py0, px0 = clamp_box(cy, cx)
    ey0, ex0 = clamp_box(ey, ex)
    return {"z": z_mid, "puncta": (py0, px0), "empty": (ey0, ex0)}


def render_xy(vol: np.ndarray, vmin: float, vmax: float, out_png: Path, title: str):
    mip = vol.max(axis=0)
    fig, ax = plt.subplots(figsize=(6, 6 * mip.shape[0] / mip.shape[1]))
    ax.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def render_ortho(vol: np.ndarray, zoom: dict, vmin: float, vmax: float, out_png: Path, title: str):
    z, (py0, px0), (ey0, ex0) = zoom["z"], zoom["puncta"], zoom["empty"]
    half = ZOOM_SIZE

    xz = vol.max(axis=1)  # (z, x)
    yz = vol.max(axis=2)  # (z, y)
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


def find_psfgen_figure(sweep_root: Path, stage_dir: Path, variant_name: str) -> Path | None:
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
    result = run(sweep_root, sweep_root / "variants.json", render_dir)

    vols = result["vols"]
    masks = result["masks"]
    variants = result["variants"]

    # Shared display range per channel, taken from the 'prod' arm's
    # percentiles (0.1-99.95) so every variant/arm is shown identically.
    display_range = {}
    for ch in CHANNELS:
        prod_vol = vols[("prod", ch)]
        nonzero = prod_vol[prod_vol > 0]
        vmin = float(np.percentile(nonzero, 0.5)) if nonzero.size else 0.0
        vmax = float(np.percentile(nonzero, 99.95)) if nonzero.size else 1.0
        display_range[ch] = (vmin, vmax)
        print(f"[display] {ch}: vmin={vmin:.1f} vmax={vmax:.1f}")

    zoom_by_channel = {}
    for ch in CHANNELS:
        zoom_by_channel[ch] = pick_zoom_centers(vols[("nodecon", ch)], masks[ch]["cell"], masks[ch]["empty"])
        print(f"[zoom] {ch}: {zoom_by_channel[ch]}")

    psfgen_dir = render_dir / "psfgen"
    psfgen_dir.mkdir(exist_ok=True)

    manifest = {"channels": {}, "variants": [v["name"] for v in variants], "display_range": display_range}

    # psfgen QC figure is per-variant (PetaKit5D generates one back
    # projector per PSF+alpha+OTFCumThresh, not per channel), computed once
    # and attached to every channel's entry for that variant so the page
    # can look it up under whichever channel is currently selected.
    psfgen_by_variant: dict[str, str] = {}
    for v in variants:
        name = v["name"]
        fig = find_psfgen_figure(sweep_root, result["stage_dir"], name)
        if fig is not None:
            dst = psfgen_dir / f"{name}_psfgen_figure.png"
            shutil.copy(fig, dst)
            psfgen_by_variant[name] = f"psfgen/{dst.name}"

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

    manifest_path = render_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
