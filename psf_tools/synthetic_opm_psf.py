#!/usr/bin/env python3
"""Physical synthetic PSF for the OPM, in the RAW SKEWED frame decon uses.

Why a model rather than another measured PSF. Decon can only put light back
where the PSF says it went. The measured bead PSF is 81 x 31 x 31 voxels
(8 um of scan, +-1.8 um of lab z), and noisy beyond ~1e-3 of its peak, so the
faint wide tail that IS the out-of-focus haze is either cropped away or buried
in noise. A model is noise-free and can be as large as needed; fitted to the
bead so its core and first rings match, it carries the tail out to many um.

Model (a point emitter, everything in sample-space micrometres):

    PSF(k, u, v) = I_sheet(n) * h_det(x, y, z)

  raw voxel (k, u, v) = scan plane k, tilted in-plane pixel u, coverslip pixel v
  n = k*dz*sin(theta)                      distance of the emitter from the sheet
  x = v*xy,  y = u*xy*cos(theta) + k*dz,  z = u*xy*sin(theta)
                                           lab position relative to the emitter

  (theta = skew angle 60 deg; deskew's shear of dz*cos(theta)/xy px/plane is
  the t = u*xy + k*dz*cos(theta) coordinate along the plane.)

  h_det  O1's 3-D widefield intensity PSF: scalar angular spectrum of a
         circular (optionally elliptical) pupil with Zernike aberrations
         (defocus, astigmatism, coma, SPHERICAL), axis = lab z.
  I_sheet  the sheet's profile across its normal: a Gaussian core plus an
         exponential tail. The tail is excitation haze.

Because h_det is evaluated at the lab position each raw voxel samples, the
well-known OPM streak (the O1 axial lobe appears sheared in raw space) falls
out of the geometry. It is NOT a free shear parameter, so the "bead streak
slope vs deskew shear" question is answered by whether dz=0.1 fits, not assumed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from psf_tools.phase_retrieve_psf import zernike_noll

XY_UM = 0.136
THETA_DEG = 60.0
MARGIN_UM = 6.0  # lab-grid margin: the FFT grid is periodic, so a defocused ring must not wrap


@dataclass
class OPMPSFModel:
    na: float = 1.1
    n_imm: float = 1.40            # silicone immersion; held fixed
    wavelength_um: float = 0.517
    na_ratio: float = 1.0          # pupil semi-axis ratio along fy / fx (O3 tilt clips it)
    zern: dict = field(default_factory=dict)  # Noll index -> rms radians (4 defocus, 5/6 astig, 7/8 coma, 11 spherical)
    sheet_sigma_um: float = 0.6
    tail_frac: float = 0.0
    tail_len_um: float = 3.0
    lobe_amp: float = 0.0          # secondary sheet maximum (seen in the bead PSF ~1 um to one side)
    lobe_off_um: float = 1.0
    lobe_sigma_um: float = 0.5
    dz_um: float = 0.1             # scan step of the PSF stack
    flip_u: bool = False
    flip_k: bool = False
    blur_u_px: float = 0.0         # extra Gaussian blur in the raw plane (bead size, registration), pixels
    blur_v_px: float = 0.0
    center_k: float = 0.0          # sub-voxel centre offsets, voxels
    center_u: float = 0.0
    center_v: float = 0.0
    xy_um: float = XY_UM
    theta_deg: float = THETA_DEG

    def to_json(self) -> dict:
        return asdict(self)


def lab_psf(m: OPMPSFModel, extent_xy_um: tuple[float, float], extent_z_um: float, step_um: float = 0.068):
    """O1 3-D intensity PSF on a lab grid, (z, y, x), axis-0 centred. Returns
    (volume, step, origin_index) where the emitter sits at index
    (nz//2, ny//2, nx//2). Intensity is normalised to its own maximum.

    Pupil: circular aperture of radius NA/lambda in spatial frequency, scalar
    angular-spectrum defocus exp(i*kz*z), kz = 2*pi*sqrt((n/lambda)^2 - f^2).
    The FFT grid spacing is `step_um`, Nyquist well above the 2NA/lambda
    cutoff; the field is band-limited so it is smooth enough to interpolate."""
    ny = int(np.ceil(extent_xy_um[1] / step_um)) * 2 + 1
    nx = int(np.ceil(extent_xy_um[0] / step_um)) * 2 + 1
    nz = int(np.ceil(extent_z_um / step_um)) * 2 + 1
    fy = np.fft.fftfreq(ny, d=step_um)[:, None]
    fx = np.fft.fftfreq(nx, d=step_um)[None, :]
    fc = m.na / m.wavelength_um
    rho = np.sqrt((fx / fc) ** 2 + (fy / (fc * m.na_ratio)) ** 2)
    support = rho <= 1.0
    theta = np.arctan2(fy / m.na_ratio, fx + 0 * fy)
    phase = np.zeros((ny, nx))
    for j, coeff in m.zern.items():
        phase = phase + coeff * zernike_noll(int(j), np.minimum(rho, 1.0), theta)
    pupil = support * np.exp(1j * phase)
    kz = 2 * np.pi * np.sqrt(np.clip((m.n_imm / m.wavelength_um) ** 2 - fx**2 - fy**2, 0, None))
    zs = (np.arange(nz) - nz // 2) * step_um
    vol = np.empty((nz, ny, nx), np.float32)
    for i, z in enumerate(zs):
        field_ = np.fft.fftshift(np.fft.ifft2(pupil * np.exp(1j * kz * z) * support))
        vol[i] = np.abs(field_) ** 2
    vol /= vol.max()
    return vol, step_um


def sheet_profile(m: OPMPSFModel, n_um: np.ndarray) -> np.ndarray:
    g = np.exp(-(n_um**2) / (2 * m.sheet_sigma_um**2))
    t = np.exp(-np.abs(n_um) / m.tail_len_um)
    lobe = np.exp(-((n_um - m.lobe_off_um) ** 2) / (2 * m.lobe_sigma_um**2))
    return (1 - m.tail_frac - m.lobe_amp) * g + m.tail_frac * t + m.lobe_amp * lobe


def raw_psf(m: OPMPSFModel, shape: tuple[int, int, int], lab=None, supersample: int = 2) -> np.ndarray:
    """The PSF on the raw skewed lattice, shape (nk, nu, nv), emitter at the
    array centre (plus the model's sub-voxel centre offsets), sum-normalised.

    `supersample` averages that many sub-positions per camera pixel along u and
    v (the pixel integrates 0.136 um, comparable to the PSF core)."""
    nk, nu, nv = shape
    th = np.radians(m.theta_deg)
    k = (np.arange(nk) - nk // 2 + m.center_k) * (-1 if m.flip_k else 1)
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5
    u = ((np.arange(nu) - nu // 2 + m.center_u)[:, None] + sub[None, :]).ravel() * (-1 if m.flip_u else 1)
    v = ((np.arange(nv) - nv // 2 + m.center_v)[:, None] + sub[None, :]).ravel()
    x = v * m.xy_um
    ext_xy = (np.abs(x).max() + MARGIN_UM, np.abs(u).max() * m.xy_um * np.cos(th) + np.abs(k).max() * m.dz_um + MARGIN_UM)
    ext_z = np.abs(u).max() * m.xy_um * np.sin(th) + MARGIN_UM
    if lab is None:
        lab, step = lab_psf(m, ext_xy, ext_z)
    else:
        lab, step = lab
    nz_, ny_, nx_ = lab.shape
    out = np.empty((nk, len(u), len(v)), np.float32)
    zc, yc, xc = nz_ // 2, ny_ // 2, nx_ // 2
    U, V = np.meshgrid(u, v, indexing="ij")
    zz = U * m.xy_um * np.sin(th) / step + zc
    xx = V * m.xy_um / step + xc
    for i in range(nk):
        yy = (U * m.xy_um * np.cos(th) + k[i] * m.dz_um) / step + yc
        h = ndi.map_coordinates(lab, [zz, yy, xx], order=1, mode="constant", cval=0.0)
        n_um = k[i] * m.dz_um * np.sin(th)   # signed: the side lobe is one-sided
        out[i] = h * sheet_profile(m, np.array(n_um))
    out = out.reshape(nk, nu, supersample, nv, supersample).mean(axis=(2, 4))
    if m.blur_u_px > 0 or m.blur_v_px > 0:
        out = ndi.gaussian_filter(out, sigma=(0, m.blur_u_px, m.blur_v_px), mode="constant")
    return out / out.sum()


def save_psf(path: Path, psf: np.ndarray, m: OPMPSFModel, extra: dict | None = None) -> None:
    import tifffile

    tifffile.imwrite(path, psf.astype(np.float32), imagej=True, metadata={"spacing": m.dz_um, "unit": "um"})
    Path(path).with_suffix(".model.json").write_text(json.dumps({**m.to_json(), **(extra or {})}, indent=1))
