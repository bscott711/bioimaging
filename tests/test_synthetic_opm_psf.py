"""synthetic_opm_psf: the model's geometry is the OPM's, and the knobs do what their names say."""
from dataclasses import replace

import numpy as np

from psf_tools import synthetic_opm_psf as S
from psf_tools.fit_synthetic_psf import PARAMS, residuals, to_model
from psf_tools.make_synthetic_psfs import recentre, taper

SHAPE = (41, 31, 21)  # (k, u, v)


def _base(**kw):
    # ideal optics, a thin sheet with no tail or lobe: the cleanest case for geometry
    return S.OPMPSFModel(na=0.8, wavelength_um=0.517, sheet_sigma_um=0.5, tail_frac=0.0, lobe_amp=0.0, **kw)


def test_normalised_finite_and_peak_at_centre():
    p = S.raw_psf(_base(), SHAPE)
    assert p.shape == SHAPE and np.isfinite(p).all() and (p >= 0).all()
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert np.unravel_index(p.argmax(), p.shape) == (20, 15, 10)


def test_no_aberration_is_point_symmetric_and_spherical_aberration_breaks_it():
    sym = S.raw_psf(_base(), SHAPE)
    # sheet symmetric in n, O1 PSF symmetric in z: reflecting (k, u) through the emitter is a symmetry
    assert np.allclose(sym, sym[::-1, ::-1, :], rtol=0, atol=2e-3 * sym.max())
    sa = S.raw_psf(_base(zern={11: 1.2}), SHAPE)
    assert not np.allclose(sa, sa[::-1, ::-1, :], rtol=0, atol=2e-3 * sa.max())  # negative control


def _slope_u_per_k(p):
    k, u, v = np.nonzero(p > 0.05 * p.max())
    w = p[k, u, v]
    kk, uu = k - np.average(k, weights=w), u - np.average(u, weights=w)
    return float(np.average(kk * uu, weights=w) / np.average(kk * kk, weights=w))


def test_detection_lobe_streak_slope_follows_the_geometry_and_the_scan_step():
    """With a very thick sheet only the O1 axial lobe shapes the PSF; it lies on y = 0, i.e. u*xy*cos(theta) = -k*dz, so
    du/dk = -dz / (xy cos theta): 1.47 px/plane at dz 0.1 (and 3x steeper at 0.3, the negative control)."""
    thick = dict(sheet_sigma_um=40.0)
    shape = (41, 61, 21)
    m1 = replace(_base(dz_um=0.1), **thick)
    s1 = _slope_u_per_k(S.raw_psf(m1, shape))
    assert abs(abs(s1) - 0.1 / (0.136 * np.cos(np.radians(60)))) < 0.2
    s3 = _slope_u_per_k(S.raw_psf(replace(m1, dz_um=0.3), shape))
    assert abs(s3) > 2.2 * abs(s1)
    # flipping u mirrors the streak's direction
    sf = _slope_u_per_k(S.raw_psf(replace(m1, flip_u=True), shape))
    assert np.sign(sf) == -np.sign(s1)


def test_sheet_tail_puts_energy_far_from_focus():
    near = S.raw_psf(_base(), (81, 31, 21))
    tail = S.raw_psf(replace(_base(), tail_frac=0.3, tail_len_um=3.0), (81, 31, 21))
    far = lambda p: float(p[:20].sum() + p[-20:].sum())
    assert far(tail) > 3 * far(near)


def test_recentre_and_taper():
    a = np.zeros((9, 9, 9), np.float32)
    a[2, 3, 6] = 1.0
    assert np.unravel_index(recentre(a).argmax(), a.shape) == (4, 4, 4)
    t = taper(np.ones((40, 40, 40)))
    assert t[0].max() == 0 and t[:, 0].max() == 0 and t[:, :, -1].max() == 0 and t[20, 20, 20] == 1.0


def test_fit_residuals_vanish_at_the_truth_and_grow_when_na_is_wrong():
    base = S.OPMPSFModel(wavelength_um=0.517, dz_um=0.1)
    x_true = np.array([p[3] for p in PARAMS], float)
    names = [p[0] for p in PARAMS]
    x_true[names.index("na")] = 0.8
    x_true[names.index("blur_u_px")] = 0.0
    x_true[names.index("blur_v_px")] = 0.0
    x_true[names.index("tail_frac")] = 0.1
    x_true[names.index("lobe_amp")] = 0.0
    meas = S.raw_psf(to_model(x_true, base), SHAPE)
    valid = np.ones(SHAPE, bool)
    at_truth = np.sqrt(np.mean(residuals(x_true, meas, base, valid) ** 2))
    x_bad = x_true.copy()
    x_bad[names.index("na")] = 0.5
    assert at_truth < 1e-3
    assert np.sqrt(np.mean(residuals(x_bad, meas, base, valid) ** 2)) > 20 * at_truth + 0.05
