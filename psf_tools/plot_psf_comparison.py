#!/usr/bin/env python3
"""Compare a measured bead PSF against a phase-retrieved replacement:
XY MIP, YZ MIP, and log-scaled Fourier-magnitude (OTF) MIP for each, stacked
so they're directly comparable. The OTF panel is the concrete visual
evidence of improved high-frequency support / reduced noise floor -- a
linear-scale OTF display hides essentially all of that detail, so it always
uses log1p.

Usage: uv run python -m psf_tools.plot_psf_comparison --measured averaged_psf.tif --retrieved averaged_psf_phase_retrieved.tif
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def _load_psf(path):
    data = tifffile.imread(path)
    while data.ndim > 3:
        data = data[0]
    return data.astype(np.float32)


def _otf_xy_mip(data):
    spectrum = np.abs(np.fft.fftshift(np.fft.fftn(data)))
    return np.log1p(np.max(spectrum, axis=0))


def _plot_row(axes, data, label):
    axes[0].imshow(np.max(data, axis=0), cmap="magma")
    axes[0].set_title(f"{label} XY MIP")
    axes[0].axis("off")

    axes[1].imshow(np.max(data, axis=2), cmap="magma")
    axes[1].set_title(f"{label} YZ MIP")
    axes[1].axis("off")

    axes[2].imshow(_otf_xy_mip(data), cmap="viridis")
    axes[2].set_title(f"{label} OTF (log|FFT|, XY MIP)")
    axes[2].axis("off")


def plot_comparison(measured_path, retrieved_path, out_png=None):
    measured = _load_psf(measured_path)
    retrieved = _load_psf(retrieved_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    _plot_row(axes[0], measured, "Measured")
    _plot_row(axes[1], retrieved, "Phase-Retrieved")

    retrieved_path = Path(retrieved_path)
    if out_png is None:
        out_png = retrieved_path.with_name(retrieved_path.stem + "_comparison.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_png


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--retrieved", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, default=None)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    out_png = plot_comparison(args.measured, args.retrieved, args.output_png)
    print(f"Saved PSF comparison to {out_png}")


if __name__ == "__main__":
    main()
