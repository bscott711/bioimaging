#!/usr/bin/env python3
"""Write the synthetic PSF set from a fit (fit_synthetic_psf.py).

  syn_{488,561}      production support (81 x 31 x 31): isolates the model from the support
  synL_{488,561}     large support (161 x 95 x 95): carries the tail out to +-5.6 um of lab z
  synL_sa0_{488,561} large, spherical terms (Noll 11, 22) zeroed: the aberration control

Files are (k, axis1=v coverslip, axis2=u tilted), like the measured bead PSF, sum-
normalised, with the sub-voxel centre offsets zeroed (the emitter at the array
centre). The 561 PSF rescales the 488 fit physically: wavelength, Zernike radians
by lambda_fit/lambda (constant OPD), and the excitation-side sheet by 561/488.
The outer 15 % of each axis is tapered to zero, as the measured PSF's edges are.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal.windows import tukey

from psf_tools.fit_synthetic_psf import to_model
from psf_tools.synthetic_opm_psf import OPMPSFModel, raw_psf, save_psf

EM_488, EM_561, EXC_RATIO = 0.517, 0.600, 561.0 / 488.0
SUPPORTS = {"syn": (81, 31, 31), "synL": (161, 95, 95)}


def taper(a: np.ndarray, frac: float = 0.15) -> np.ndarray:
    for ax, n in enumerate(a.shape):
        w = tukey(n, alpha=2 * frac)
        a = a * w.reshape([-1 if i == ax else 1 for i in range(a.ndim)])
    return a


def recentre(a: np.ndarray) -> np.ndarray:
    """Integer-roll the maximum to the array centre, the convention of the measured
    bead PSF (peak at (40, 15, 15)); an off-centre PSF shifts the decon output."""
    pk = np.unravel_index(a.argmax(), a.shape)
    return np.roll(a, [n // 2 - p for n, p in zip(a.shape, pk)], axis=(0, 1, 2))


def channel_model(base: OPMPSFModel, fit_wavelength: float, channel: str) -> OPMPSFModel:
    m = replace(base, center_k=0.0, center_u=0.0, center_v=0.0)
    if channel == "561":
        sc = fit_wavelength / EM_561
        m = replace(m, wavelength_um=EM_561, zern={j: c * sc for j, c in m.zern.items()},
                    sheet_sigma_um=m.sheet_sigma_um * EXC_RATIO, tail_len_um=m.tail_len_um * EXC_RATIO,
                    lobe_off_um=m.lobe_off_um * EXC_RATIO, lobe_sigma_um=m.lobe_sigma_um * EXC_RATIO)
    else:
        m = replace(m, wavelength_um=EM_488)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    d = json.loads(a.fit.read_text())
    r = d["results"][0]
    base = to_model(np.array(r["x"]), OPMPSFModel(wavelength_um=d["wavelength_um"], dz_um=d["dz"],
                                                   flip_u=r["flip_u"], flip_k=r["flip_k"]))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for key, shape in (("syn", SUPPORTS["syn"]), ("synL", SUPPORTS["synL"]), ("synL_sa0", SUPPORTS["synL"])):
        for ch in ("488", "561"):
            m = channel_model(base, d["wavelength_um"], ch)
            if key.endswith("sa0"):
                m = replace(m, zern={j: (0.0 if j in (11, 22) else c) for j, c in m.zern.items()})
            psf = taper(recentre(raw_psf(m, shape)))
            psf /= psf.sum()
            path = a.out_dir / f"{key}_{ch}.tif"
            save_psf(path, psf.transpose(0, 2, 1), m, {"fit": str(a.fit), "fit_log_rms": r["log_rms"],
                                                       "support": list(shape), "axes": "k, v(coverslip), u(tilted)"})
            index[f"{key}_{ch}"] = str(path)
            print(f"{path.name}: shape {psf.shape}, peak {np.unravel_index(psf.argmax(), psf.shape)}, "
                  f"energy beyond +-15 u px {1 - psf[:, shape[1] // 2 - 15:shape[1] // 2 + 16, :].sum() / 1:.4f}" if shape[1] > 31 else
                  f"{path.name}: shape {psf.shape}, peak {np.unravel_index(psf.argmax(), psf.shape)}")
    (a.out_dir.parent.parent / "psfs.json").write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
