import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from scipy.ndimage import fourier_shift

from psf_tools.extract_bead_psf import (
    _isolation_score,
    _mirror_symmetry_score,
    apply_quality_gates,
    bootstrap_reregister,
    robust_average,
    score_bead_crops,
)


def _add_gaussian(volume, center, amplitude, sigma):
    zg, yg, xg = np.indices(volume.shape)
    r2 = (zg - center[0]) ** 2 + (yg - center[1]) ** 2 + (xg - center[2]) ** 2
    volume += amplitude * np.exp(-r2 / (2 * sigma**2))


def _reference_extract_average_psf(data, points, radius_z, radius_xy, normalize=True):
    """Independent re-implementation of the pre-refactor extract_bead_psf.py
    algorithm (plain 3D (Z, Y, X) case only), used as a ground truth to catch
    any behavior change introduced by the modular refactor."""
    valid_crops = []
    shape = data.shape
    for pt in points:
        pt = np.round(pt).astype(int)
        z, y, x = pt
        z_min, z_max = z - radius_z, z + radius_z + 1
        y_min, y_max = y - radius_xy, y + radius_xy + 1
        x_min, x_max = x - radius_xy, x + radius_xy + 1
        if (z_min < 0 or z_max > shape[0] or y_min < 0 or y_max > shape[1]
                or x_min < 0 or x_max > shape[2]):
            continue
        crop = data[z_min:z_max, y_min:y_max, x_min:x_max]

        crop_float = crop.astype(float)
        bg = np.percentile(crop_float, 10)
        crop_sub = np.clip(crop_float - bg, 0, None)

        Z, Y, X = crop_sub.shape
        z0, y0, x0 = Z / 2.0 - 0.5, Y / 2.0 - 0.5, X / 2.0 - 0.5
        z_grid, y_grid, x_grid = np.ogrid[0:Z, 0:Y, 0:X]

        weight = 0.0
        for _ in range(10):
            window_mask = (
                (z_grid - z0) ** 2 / 5**2 + (y_grid - y0) ** 2 / 5**2 + (x_grid - x0) ** 2 / 5**2
            ) <= 1
            windowed = crop_sub * window_mask
            weight = windowed.sum()
            if weight == 0:
                break
            z0 = (z_grid * windowed).sum() / weight
            y0 = (y_grid * windowed).sum() / weight
            x0 = (x_grid * windowed).sum() / weight
        if weight == 0:
            continue

        cov_mask = (
            (z_grid - z0) ** 2 / (radius_z * 0.5) ** 2
            + (y_grid - y0) ** 2 / (radius_xy * 0.5) ** 2
            + (x_grid - x0) ** 2 / (radius_xy * 0.5) ** 2
        ) <= 1
        windowed_cov = crop_sub * cov_mask
        weight_cov = windowed_cov.sum()
        if weight_cov > 0:
            dz, dy, dx = z_grid - z0, y_grid - y0, x_grid - x0
            c_zz = (dz**2 * windowed_cov).sum() / weight_cov
            c_yy = (dy**2 * windowed_cov).sum() / weight_cov
            c_xx = (dx**2 * windowed_cov).sum() / weight_cov
            c_zy = (dz * dy * windowed_cov).sum() / weight_cov
            c_zx = (dz * dx * windowed_cov).sum() / weight_cov
            c_yx = (dy * dx * windowed_cov).sum() / weight_cov
            cov_matrix = np.array([[c_zz, c_zy, c_zx], [c_zy, c_yy, c_yx], [c_zx, c_yx, c_xx]])
        else:
            cov_matrix = np.eye(3)

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except Exception:
            inv_cov = np.eye(3)

        dz, dy, dx = z_grid - z0, y_grid - y0, x_grid - x0
        mahal_sq = (
            inv_cov[0, 0] * dz**2 + inv_cov[1, 1] * dy**2 + inv_cov[2, 2] * dx**2
            + 2 * inv_cov[0, 1] * dz * dy + 2 * inv_cov[0, 2] * dz * dx + 2 * inv_cov[1, 2] * dy * dx
        )
        mahal = np.sqrt(np.clip(mahal_sq, 0, None))
        soft_mask = np.clip((5.0 - mahal) / 2.0, 0, 1)

        masked_crop = crop_sub * soft_mask
        center_target = np.array([Z / 2 - 0.5, Y / 2 - 0.5, X / 2 - 0.5])
        shift_vec = center_target - np.array([z0, y0, x0])
        img_fft = np.fft.fftn(masked_crop)
        img_shifted_fft = fourier_shift(img_fft, shift_vec)
        final_crop = np.fft.ifftn(img_shifted_fft).real
        final_crop[final_crop < 0] = 0

        if normalize:
            energy = np.sum(final_crop)
            if energy > 0:
                final_crop = final_crop / energy
            else:
                continue

        valid_crops.append(final_crop)

    psf_stack = np.stack(valid_crops, axis=0)
    return np.mean(psf_stack, axis=0)


@pytest.fixture
def two_bead_volume():
    rng = np.random.default_rng(1)
    vol = 5.0 + rng.normal(0, 0.3, size=(40, 40, 40))
    vol = np.clip(vol, 0, None)
    _add_gaussian(vol, (15, 10, 10), 400, 2.0)
    _add_gaussian(vol, (20, 25, 25), 350, 1.8)
    points = np.array([[15, 10, 10], [20, 25, 25]], dtype=float)
    return vol, points


def test_score_bead_crops_matches_reference_implementation(two_bead_volume):
    """The refactor into helper functions must not change any numeric result:
    score_bead_crops + robust_average(combine='mean') must reproduce the
    original inline algorithm exactly (this is the auto_reject=False,
    bootstrap_reregister_beads=False, combine='mean', apodize_tails=False
    backward-compat path)."""
    vol, points = two_bead_volume
    radius_z, radius_xy = 7, 7

    reference = _reference_extract_average_psf(vol, points, radius_z, radius_xy, normalize=True)

    scores = score_bead_crops(vol, points, radius_z, radius_xy, normalize=True)
    assert len(scores) == 2
    assert all(s["accepted"] for s in scores)  # score_bead_crops defaults to accepted=True

    new_avg = robust_average([s["final_crop"] for s in scores], combine="mean")
    assert np.allclose(reference, new_avg, atol=1e-10)


def test_quality_gates_reject_overlapping_and_low_snr_beads():
    """Synthetic volume with one clean isolated bead, one bead too dim to
    trust, and a pair of beads close enough to be overlapping crops. Only
    the clean bead should survive default quality gates."""
    rng = np.random.default_rng(42)
    shape = (50, 50, 50)
    vol = 5.0 + rng.normal(0, 0.5, size=shape)
    vol = np.clip(vol, 0, None)

    _add_gaussian(vol, (25, 10, 10), 500, 2.0)   # clean, isolated, bright
    _add_gaussian(vol, (25, 10, 40), 0.5, 2.0)   # far too dim (low SNR)
    _add_gaussian(vol, (25, 30, 10), 500, 2.0)   # overlapping pair --
    _add_gaussian(vol, (25, 30, 16), 500, 2.0)   # -- 6px apart, radius_xy=8

    points = np.array([
        [25, 10, 10],
        [25, 10, 40],
        [25, 30, 10],
        [25, 30, 16],
    ], dtype=float)

    radius_z = radius_xy = 8
    scores = score_bead_crops(vol, points, radius_z, radius_xy, normalize=True)
    assert len(scores) == 4

    apply_quality_gates(scores)
    accepted = [s["index"] for s in scores if s["accepted"]]
    assert accepted == [0]


def test_mirror_symmetry_score_distinguishes_symmetric_from_asymmetric():
    y, x = np.mgrid[0:21, 0:21]
    symmetric_slice = np.exp(-((y - 10) ** 2 + (x - 10) ** 2) / (2 * 3.0**2))
    symmetric_crop = np.stack([symmetric_slice * 0.2, symmetric_slice, symmetric_slice * 0.3])
    assert _mirror_symmetry_score(symmetric_crop) > 0.99

    asymmetric_slice = np.full((21, 21), 0.01)
    asymmetric_slice[:10, :10] = 10.0  # bright in one quadrant only
    asymmetric_crop = np.stack([asymmetric_slice * 0.2, asymmetric_slice, asymmetric_slice * 0.3])
    assert _mirror_symmetry_score(asymmetric_crop) < 0.5


def test_isolation_score_is_anisotropic_nearest_neighbor_distance():
    points = np.array([[0, 0, 0], [0, 0, 5], [0, 10, 10]], dtype=float)
    radius_z, radius_xy = 8, 8
    # nearest neighbor to point 0 is point 1, 5px away in X: 5/8
    assert _isolation_score(points, 0, radius_z, radius_xy) == pytest.approx(5 / 8)
    # nearest neighbor to point 2 is point 1: dy=10/8, dx=5/8
    expected = np.sqrt((10 / 8) ** 2 + (5 / 8) ** 2)
    assert _isolation_score(points, 2, radius_z, radius_xy) == pytest.approx(expected)


def test_apply_quality_gates_thresholds_each_metric_independently():
    """Directly exercise the gate logic with hand-built scores, independent
    of any particular synthetic-volume fixture."""
    scores = [
        {"isolation": 3.0, "snr": 50.0, "mirror_symmetry": 0.9, "meanshift_residual_px": 0.01},  # good
        {"isolation": 0.5, "snr": 50.0, "mirror_symmetry": 0.9, "meanshift_residual_px": 0.01},  # too close
        {"isolation": 3.0, "snr": 2.0, "mirror_symmetry": 0.9, "meanshift_residual_px": 0.01},   # too dim
        {"isolation": 3.0, "snr": 50.0, "mirror_symmetry": 0.1, "meanshift_residual_px": 0.01},  # asymmetric
        {"isolation": 3.0, "snr": 50.0, "mirror_symmetry": 0.9, "meanshift_residual_px": 5.0},   # didn't converge
    ]
    apply_quality_gates(scores)
    assert [s["accepted"] for s in scores] == [True, False, False, False, False]


def test_bootstrap_reregister_improves_alignment_to_shifted_crop():
    """A single off-center crop among several well-centered ones should land
    closer to the clean pattern after re-registration against the (robust,
    outlier-dominated-by-the-majority) median template."""
    y, x = np.mgrid[0:31, 0:31]

    def gaussian_at(cy, cx):
        return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 2.5**2))

    clean = np.stack([gaussian_at(15, 15)] * 5)
    shifted = np.stack([gaussian_at(15.6, 14.7)] * 5)
    crops = [clean, clean, clean, clean, clean, shifted]

    registered = bootstrap_reregister(crops, upsample_factor=20)
    resid_before = np.linalg.norm(shifted - clean)
    resid_after = np.linalg.norm(registered[-1] - clean)
    assert resid_after < resid_before * 0.1


def test_robust_average_combine_modes():
    crops = [np.full((3, 3, 3), v, dtype=float) for v in (1.0, 2.0, 3.0, 100.0)]
    assert np.allclose(robust_average(crops, combine="mean"), 26.5)
    assert np.allclose(robust_average(crops, combine="median"), 2.5)
    trimmed = robust_average(crops, combine="trimmed_mean", trim_fraction=0.25)
    assert np.all(trimmed < 26.5)  # trimming the 100.0 outlier pulls the mean down

    with pytest.raises(ValueError):
        robust_average(crops, combine="not_a_real_method")
