#!/usr/bin/env python3
"""Fit `synthetic_opm_psf.OPMPSFModel` to a measured, sum-normalised bead PSF.

Loss: log-ratio of (model + f) to (data + f) over every voxel the extraction
left non-zero (its edge planes are zeroed), f a few noise sigmas. Log space so
the faint tail -- the part decon needs for haze -- counts as much as the core;
the floor f keeps noise from dominating. Four discrete orientation choices
(flip_u, flip_k) are each fitted and the lowest loss wins; dz is a fit-time
option so the open "is dz 0.1 right?" question can be answered by the loss.

  uv run python -m psf_tools.fit_synthetic_psf --psf <bead.tif> --out fit.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import least_squares

from psf_tools.synthetic_opm_psf import OPMPSFModel, raw_psf

FLOOR = 1.0e-6  # ~10x the measured noise sigma (1.1e-7)
TAIL_WEIGHT = 0.4
CORE_WEIGHT = 25.0

# name, lo, hi, start
PARAMS = [
    ("na", 0.25, 1.35, 0.7),
    ("na_ratio", 0.4, 1.8, 1.0),
    ("z4", -3.0, 3.0, 0.0),    # defocus
    ("z5", -3.0, 3.0, 0.0),    # astigmatism
    ("z6", -3.0, 3.0, 0.0),
    ("z7", -3.0, 3.0, 0.0),    # coma
    ("z8", -3.0, 3.0, 0.0),
    ("z9", -2.0, 2.0, 0.0),    # trefoil
    ("z10", -2.0, 2.0, 0.0),
    ("z11", -4.0, 4.0, 0.0),   # spherical
    ("z12", -2.0, 2.0, 0.0),   # secondary astigmatism
    ("z13", -2.0, 2.0, 0.0),
    ("z22", -2.0, 2.0, 0.0),   # secondary spherical
    ("sheet_sigma_um", 0.15, 2.5, 0.6),
    ("tail_frac", 0.0, 0.9, 0.05),
    ("tail_len_um", 0.5, 12.0, 3.0),
    ("lobe_amp", 0.0, 0.25, 0.02),
    ("lobe_off_um", -3.0, 3.0, -1.0),
    ("lobe_sigma_um", 0.15, 1.5, 0.4),
    ("blur_u_px", 0.0, 2.0, 0.3),
    ("blur_v_px", 0.0, 2.0, 0.3),
    ("center_k", -2.0, 2.0, 0.0),
    ("center_u", -2.0, 2.0, 0.0),
    ("center_v", -2.0, 2.0, 0.0),
]


def to_model(x, base: OPMPSFModel) -> OPMPSFModel:
    d = dict(zip((p[0] for p in PARAMS), x))
    zern = {int(k[1:]): v for k, v in d.items() if k.startswith("z")}
    kw = {k: v for k, v in d.items() if not k.startswith("z")}
    return replace(base, zern=zern, **kw)


def load_measured(path: Path) -> np.ndarray:
    """(k, u, v) sum-normalised. The saved PSF is (k, axis1, axis2) with axis2 the
    sheet-tilted axis (walk-method drift: +1.05 px/plane on axis2, ~0 on axis1)."""
    a = tifffile.imread(path).astype(np.float64)
    a = a.transpose(0, 2, 1)
    return a / a.sum()


def residuals(x, meas, base, valid):
    m = to_model(x, base)
    mod = raw_psf(m, meas.shape)
    d = np.clip(meas, 0, None)
    tail = (np.log(mod + FLOOR) - np.log(d + FLOOR))[valid]
    # core term: in sqrt-intensity relative to the peak, so the bright voxels (which the
    # log loss, dominated by the 84 % of voxels below 1e-5, would trade away) still count
    core = (np.sqrt(mod) - np.sqrt(d))[valid] / np.sqrt(d.max()) * CORE_WEIGHT
    return np.concatenate([tail * TAIL_WEIGHT, core])


def diagnostics(x, meas, base) -> dict:
    """Unweighted numbers for comparing fits: log-rms over every valid voxel, and the
    relative error of the bright core (voxels above 5 % of the peak)."""
    valid = np.broadcast_to((meas.max(axis=(1, 2)) > 0)[:, None, None], meas.shape)
    mod = raw_psf(to_model(x, base), meas.shape)
    d = np.clip(meas, 0, None)
    lr = (np.log(mod + FLOOR) - np.log(d + FLOOR))[valid]
    core = (d > 0.05 * d.max()) & valid
    return {"log_rms": float(np.sqrt(np.mean(lr**2))),
            "core_rel_err": float(np.sqrt(np.mean(((mod - d)[core] / d.max()) ** 2))),
            "core_voxels": int(core.sum())}


def fit(meas: np.ndarray, base: OPMPSFModel, starts: int = 3, seed: int = 0, verbose: int = 0):
    valid = np.broadcast_to((meas.max(axis=(1, 2)) > 0)[:, None, None], meas.shape)
    lo = np.array([p[1] for p in PARAMS])
    hi = np.array([p[2] for p in PARAMS])
    x0 = np.array([p[3] for p in PARAMS])
    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        x_start = x0 if s == 0 else np.clip(x0 + rng.normal(0, 0.35, x0.size) * (hi - lo) * 0.3, lo, hi)
        r = least_squares(residuals, x_start, bounds=(lo, hi), args=(meas, base, valid),
                          x_scale=(hi - lo) / 4, diff_step=1e-2, max_nfev=120, verbose=verbose)
        cost = float(np.sqrt(np.mean(r.fun**2)))
        if best is None or cost < best[0]:
            best = (cost, r.x)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dz", type=float, default=0.1)
    ap.add_argument("--wavelength-um", type=float, default=0.515)
    ap.add_argument("--starts", type=int, default=3)
    a = ap.parse_args(argv)
    meas = load_measured(a.psf)
    results = []
    # flipping u and k together is a symmetry (point reflection) -> only two distinct orientations
    for fu, fk in ((False, False), (True, False)):
        base = OPMPSFModel(wavelength_um=a.wavelength_um, dz_um=a.dz, flip_u=fu, flip_k=fk)
        cost, x = fit(meas, base, a.starts)
        dg = diagnostics(x, meas, base)
        results.append({"flip_u": fu, "flip_k": fk, "cost": cost, "x": x.tolist(), **dg})
        print(f"dz={a.dz} flip_u={fu} flip_k={fk}: cost {cost:.4f} {dg}", flush=True)
    results.sort(key=lambda r: r["cost"])
    a.out.write_text(json.dumps({"dz": a.dz, "wavelength_um": a.wavelength_um, "params": [p[0] for p in PARAMS],
                                 "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
