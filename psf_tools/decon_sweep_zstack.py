#!/usr/bin/env python3
"""Z-stack scrub videos for the decon sweep artifact's Z-stack tab.

Publishing one file per Z plane is impossible at this scale (20 variants x
2 channels x 419 planes, against the artifact's total 255-file / 64MB
budget for the whole page), and a JPEG sprite sheet at a resolution worth
looking at would blow the size budget on its own (measured: a single
465x1458, CRF-32 attempt at naive per-frame JPEG-equivalent encoding
comes in far larger than the budget allows). Adjacent Z planes in a real
volume are highly correlated, so inter-frame video compression (VP9) is
the only encoding that fits: one small silent .webm per variant per
channel, with every real Z plane as a frame (no decimation), scrubbed via
a <input type=range> driving `video.currentTime` in the JS.

Measured on this dataset: 2x spatial downscale + CRF 45 keeps every
variant/channel under ~3MB (worst case is C1, the low-SNR channel, whose
tight display range stretches small noise fluctuations across more of
the 0-255 range than C0's -- confirmed empirically, not assumed), for an
estimated ~25-30MB total across all 40 videos.

Display range is computed once per channel from the `prod` arm, exactly
like decon_sweep_quickrender.py's MIP renders, so a Z-stack video's
intensity mapping is identical to (and thus comparable with) the MIP/ortho
views elsewhere in the artifact.

Usage: uv run python -m psf_tools.decon_sweep_zstack --sweep-root <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

from psf_tools.decon_sweep_report import CHANNELS, find_dsr_tif

DOWNSCALE = 2
FPS = 24
VP9_PARAMS = ["-deadline", "realtime", "-cpu-used", "8", "-crf", "45", "-b:v", "0"]


def encode_zstack(vol: np.ndarray, vmin: float, vmax: float, out_path: Path) -> None:
    norm = np.clip((vol - vmin) / (vmax - vmin), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    small = u8[:, ::DOWNSCALE, ::DOWNSCALE]
    # VP9/yuv420p wants 3 channels; luma alone carries all the (grayscale)
    # detail so replicating instead of colorizing costs nothing real.
    frames = np.repeat(small[..., None], 3, axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path, frames, fps=FPS, codec="libvpx-vp9", pixelformat="yuv420p",
        output_params=VP9_PARAMS,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, required=True)
    args = ap.parse_args()

    sweep_root = args.sweep_root
    cfg = json.loads((sweep_root / "variants.json").read_text())
    stage_dir = Path(cfg["stage_dir"])
    variants = [v for v in cfg["variants"] if not v.get("failed")]

    out_dir = sweep_root / "render" / "zstack"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    for ch in CHANNELS:
        prod_path = find_dsr_tif(sweep_root, stage_dir, "prod", True, ch)
        prod_vol = tifffile.imread(prod_path).astype(np.float32)
        nonzero = prod_vol[prod_vol > 0]
        vmin = float(np.percentile(nonzero, 0.5)) if nonzero.size else 0.0
        vmax = float(np.percentile(nonzero, 99.95)) if nonzero.size else 1.0
        print(f"[zstack] {ch}: vmin={vmin:.1f} vmax={vmax:.1f}")

        for v in variants:
            name = v["name"]
            decon = bool(v.get("decon", False))
            path = find_dsr_tif(sweep_root, stage_dir, name, decon, ch)
            vol = tifffile.imread(path).astype(np.float32)
            out_path = out_dir / f"z_{name}_{ch}.webm"
            encode_zstack(vol, vmin, vmax, out_path)
            manifest.setdefault(ch, {})[name] = {
                "path": f"zstack/{out_path.name}",
                "n_z": int(vol.shape[0]),
                "fps": FPS,
            }
            size_mb = out_path.stat().st_size / 1e6
            print(f"[zstack] wrote {out_path.name} ({size_mb:.2f} MB)")

    manifest_path = sweep_root / "render" / "zstack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
