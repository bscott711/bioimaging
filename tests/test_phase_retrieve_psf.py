import numpy as np
import pytest

from psf_tools.phase_retrieve_psf import (
    PSFModelParams,
    _pupil_polar_grid,
    _read_dz_from_tif,
    default_wavelength_um_for_name,
    noll_to_zern,
    pupil_support_mask,
    regenerate_and_save_psf,
    retrieve_pupil_phase,
    simulate_psf_from_pupil,
    zernike_noll,
    PhaseRetrievalResult,
)

NOLL_INDICES = (4, 5, 6, 7, 8, 9, 10, 11)


@pytest.fixture
def params():
    return PSFModelParams(na=1.35, wavelength_um=0.52, refractive_index=1.518,
                           xy_pixel_size_um=0.136, dz_um=0.1)


def _ground_truth_pupil(params, ny, nx, coeffs):
    support = pupil_support_mask(ny, nx, params)
    rho, theta = _pupil_polar_grid(ny, nx, params)
    phase = np.zeros((ny, nx))
    for j, c in coeffs.items():
        phase += c * zernike_noll(j, rho, theta)
    return support.astype(complex) * np.exp(1j * phase)


def _ncc(a, b):
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))


def test_zernike_noll_orthonormality():
    n = 200
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    mask = rho <= 1.0
    area = mask.sum()

    basis = {j: zernike_noll(j, rho[mask], theta[mask]) for j in (1, 2, 3, *NOLL_INDICES)}
    max_offdiag = 0.0
    diag_values = []
    for i in basis:
        for j in basis:
            inner = np.sum(basis[i] * basis[j]) / area
            if i == j:
                diag_values.append(inner)
            else:
                max_offdiag = max(max_offdiag, abs(inner))

    assert max_offdiag < 0.05
    assert all(0.8 < d < 1.2 for d in diag_values)


def test_noll_to_zern_matches_standard_ordering():
    # Piston, tip, tilt, defocus, astigmatism x2, coma x2, trefoil x2, spherical
    assert noll_to_zern(1) == (0, 0)
    assert noll_to_zern(4) == (2, 0)
    assert noll_to_zern(11) == (4, 0)


def test_phase_retrieval_recovers_pupil_noise_free(params):
    ny = nx = 48
    nz = 41
    gt_coeffs = {4: 0.4, 5: 0.2, 6: -0.15, 11: 0.3}
    pupil_gt = _ground_truth_pupil(params, ny, nx, gt_coeffs)

    z_positions_um = (np.arange(nz) - nz // 2) * params.dz_um
    psf_gt = simulate_psf_from_pupil(pupil_gt, z_positions_um, params)
    psf_gt_norm = psf_gt / psf_gt.sum()

    result = retrieve_pupil_phase(psf_gt_norm, params, n_iterations=40,
                                   zernike_regularize=True, noll_indices=NOLL_INDICES)

    for j in NOLL_INDICES:
        expected = gt_coeffs.get(j, 0.0)
        assert result.zernike_coeffs[j] == pytest.approx(expected, abs=0.02)

    psf_regen = simulate_psf_from_pupil(result.pupil, z_positions_um, params)
    assert _ncc(psf_regen, psf_gt) > 0.999


def test_phase_retrieval_recovers_pupil_with_noise(params):
    ny = nx = 48
    nz = 41
    gt_coeffs = {4: 0.4, 5: 0.2, 6: -0.15, 11: 0.3}
    pupil_gt = _ground_truth_pupil(params, ny, nx, gt_coeffs)

    z_positions_um = (np.arange(nz) - nz // 2) * params.dz_um
    psf_gt = simulate_psf_from_pupil(pupil_gt, z_positions_um, params)
    psf_gt_scaled = psf_gt / psf_gt.max() * 2000.0  # realistic bead peak ~2000 counts

    rng = np.random.default_rng(0)
    psf_noisy = rng.poisson(psf_gt_scaled).astype(float)
    psf_noisy_norm = psf_noisy / psf_noisy.sum()

    result = retrieve_pupil_phase(psf_noisy_norm, params, n_iterations=40,
                                   zernike_regularize=True, noll_indices=NOLL_INDICES)

    for j in NOLL_INDICES:
        expected = gt_coeffs.get(j, 0.0)
        assert result.zernike_coeffs[j] == pytest.approx(expected, abs=0.05)

    psf_regen = simulate_psf_from_pupil(result.pupil, z_positions_um, params)
    assert _ncc(psf_regen, psf_gt) > 0.99


def test_phase_retrieval_zero_aberration_stays_zero(params):
    ny = nx = 32
    nz = 31
    pupil_gt = _ground_truth_pupil(params, ny, nx, {})  # no aberration
    z_positions_um = (np.arange(nz) - nz // 2) * params.dz_um
    psf_gt = simulate_psf_from_pupil(pupil_gt, z_positions_um, params)
    psf_gt_norm = psf_gt / psf_gt.sum()

    result = retrieve_pupil_phase(psf_gt_norm, params, n_iterations=30,
                                   zernike_regularize=True, noll_indices=NOLL_INDICES)

    for j in NOLL_INDICES:
        assert abs(result.zernike_coeffs[j]) < 0.01


def test_psf_tiff_roundtrip_preserves_dz_tag(params, tmp_path):
    ny = nx = 16
    support = pupil_support_mask(ny, nx, params)
    pupil = support.astype(complex)
    result = PhaseRetrievalResult(
        pupil=pupil, zernike_coeffs={4: 0.1}, noll_indices=(4,),
        residual_history=[0.01], params=params,
    )

    out_tif = tmp_path / "phase_retrieved_psf.tif"
    out_json = tmp_path / "phase_retrieved_psf.json"
    psf = regenerate_and_save_psf(result, nz=11, out_tif_path=out_tif,
                                   out_json_path=out_json, target_energy=1.0)

    assert psf.shape == (11, ny, nx)
    assert out_tif.exists() and out_json.exists()
    assert _read_dz_from_tif(out_tif) == pytest.approx(params.dz_um)


def test_default_wavelength_lookup_uses_emission_not_excitation_table():
    # "YG" beads -> yellow-green emission default, not the 488nm excitation label
    wl = default_wavelength_um_for_name("YGBeads_PDMS_PSFmeasurement")
    assert wl is not None
    assert 0.49 < wl < 0.53

    assert default_wavelength_um_for_name("no_known_fluorophore_here") is None


def test_psf_model_params_rejects_na_above_refractive_index():
    with pytest.raises(ValueError):
        PSFModelParams(na=1.6, wavelength_um=0.52, refractive_index=1.33, dz_um=0.1)
