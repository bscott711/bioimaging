#!/usr/bin/env python3
"""Measured vs model PSF: log-scale MIPs and axis profiles. `--fit` is the JSON
written by fit_synthetic_psf (best orientation is used)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from psf_tools.fit_synthetic_psf import FLOOR, load_measured, to_model
from psf_tools.synthetic_opm_psf import OPMPSFModel, raw_psf


def render(psf_path: Path, fit_json: Path, out_png: Path, rank: int = 0) -> float:
    d = json.loads(fit_json.read_text())
    r = d["results"][rank]
    base = OPMPSFModel(wavelength_um=d["wavelength_um"], dz_um=d["dz"], flip_u=r["flip_u"], flip_k=r["flip_k"])
    meas = load_measured(psf_path)
    mod = raw_psf(to_model(np.array(r["x"]), base), meas.shape)
    k0, u0, v0 = np.unravel_index(meas.argmax(), meas.shape)
    fig, ax = plt.subplots(3, 4, figsize=(17, 11))
    lg = lambda a: np.log10(np.clip(a / a.max(), 1e-6, None))
    for col, (name, a) in enumerate((("measured", meas), ("model", mod))):
        ax[0, 2 * col].imshow(lg(a.max(axis=2)).T, origin="lower", vmin=-4.5, vmax=0, aspect="auto")
        ax[0, 2 * col].set_title(f"{name}: k (x) vs u (y), MIP over v"); 
        ax[0, 2 * col + 1].imshow(lg(a.max(axis=1)).T, origin="lower", vmin=-4.5, vmax=0, aspect="auto")
        ax[0, 2 * col + 1].set_title(f"{name}: k vs v, MIP over u")
        ax[1, 2 * col].imshow(lg(a.max(axis=0)), origin="lower", vmin=-4.5, vmax=0)
        ax[1, 2 * col].set_title(f"{name}: u vs v, MIP over k")
        ax[1, 2 * col + 1].imshow(lg(a[:, :, v0]).T, origin="lower", vmin=-4.5, vmax=0, aspect="auto")
        ax[1, 2 * col + 1].set_title(f"{name}: k vs u slice at v0")
    for a_, (title, mp, mm) in zip(ax[2], (("profile along k", (slice(None), u0, v0), 0), ("along u", (k0, slice(None), v0), 0),
                                           ("along v", (k0, u0, slice(None)), 0), ("plane energy vs k", None, 0))):
        if mp is None:
            a_.semilogy(meas.sum(axis=(1, 2)), "k", label="measured"); a_.semilogy(mod.sum(axis=(1, 2)), "r", label="model")
        else:
            a_.semilogy(np.clip(meas[mp], 1e-9, None), "k", label="measured"); a_.semilogy(np.clip(mod[mp], 1e-9, None), "r", label="model")
        a_.set_title(title); a_.axhline(FLOOR, color="gray", ls=":"); a_.legend(); a_.set_ylim(1e-8, None)
    fig.suptitle(f"{fit_json.name} rank{rank} flip_u={r['flip_u']} flip_k={r['flip_k']} log-rms={r['log_rms']:.3f}")
    fig.tight_layout(); fig.savefig(out_png, dpi=75); plt.close(fig)
    return r["log_rms"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=Path, required=True); ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args(); print(render(a.psf, a.fit, a.out, a.rank))
