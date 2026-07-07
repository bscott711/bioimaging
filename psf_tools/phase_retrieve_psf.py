"""Iterative phase-retrieval PSF fitting (Gerchberg-Saxton / Hanser et al. 2004).

Fits a physical pupil-function model (NA, wavelength, refractive index,
low-order Zernike aberrations) to a measured bead PSF stack -- such as the
one produced by ``psf_tools.extract_bead_psf`` -- by alternating between the
pupil plane and each z-slice's measured intensity. The converged pupil
regenerates a smooth, physically-constrained, noise-free PSF that is a
drop-in replacement for the measured one wherever ``psf_path=`` is used
(e.g. ``opym.petakit.submit_remote_decon_job``).

Physical model: scalar angular-spectrum defocus propagation of a pupil
function bounded by the objective's NA, i.e. standard wide-field through-
focus image formation. This applies directly to raw (pre-deskew) OPM bead
stacks, since deconvolution runs before deskewing in this pipeline -- the
raw stack's Z axis is the detection objective's true optical axis, not a
sheared one.

Run as a script: ``uv run python -m psf_tools.phase_retrieve_psf --help``
"""

import argparse
import json
from dataclasses import dataclass, field
from math import factorial
from pathlib import Path

import numpy as np
import tifffile

DEFAULT_NOLL_INDICES = (4, 5, 6, 7, 8, 9, 10, 11)

# Approximate emission peaks (microns) for fluorophores/beads used in this
# lab. These are DEFAULTS ONLY -- the labels in analyze_channels.FLUOROPHORES
# encode *excitation* laser lines (488/561/640/405nm), not emission, so this
# table is deliberately kept separate. Override with --wavelength-um for
# anything quantitative.
EMISSION_WAVELENGTH_UM = {
    "GFP/mNG (488)": 0.517,
    "mScarlet/CF568/TdTomato (561)": 0.600,
    "CF647/SiR (640)": 0.667,
    "Hoechst/Violet (405)": 0.455,
    "YG Beads (488 nm excitation)": 0.515,
    "Calcein Violet (cav) [Excited by 405nm]": 0.450,
}


@dataclass
class PSFModelParams:
    na: float
    wavelength_um: float
    refractive_index: float
    xy_pixel_size_um: float = 0.136
    dz_um: float = None
    zernike_noll_indices: tuple = DEFAULT_NOLL_INDICES

    def __post_init__(self):
        if self.na > self.refractive_index:
            raise ValueError(
                f"NA ({self.na}) cannot exceed the medium refractive index "
                f"({self.refractive_index}) -- NA = n*sin(theta) <= n."
            )
        if self.dz_um is not None and self.dz_um <= 0:
            raise ValueError(f"dz_um must be positive, got {self.dz_um}")


@dataclass
class PhaseRetrievalResult:
    pupil: np.ndarray
    zernike_coeffs: dict
    noll_indices: tuple
    residual_history: list
    params: PSFModelParams


# ---------------------------------------------------------------------------
# Zernike polynomials (Noll 1976 indexing), verified numerically against the
# standard orthonormality relation over the unit disk.
# ---------------------------------------------------------------------------

def noll_to_zern(j):
    """Convert a 1-based Noll index to (radial degree n, azimuthal freq m)."""
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def zernike_radial(n, m, rho):
    m = abs(m)
    R = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * factorial(n - k)) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
        R = R + c * rho ** (n - 2 * k)
    return R


def zernike_noll(j, rho, theta):
    """Orthonormal Zernike polynomial (Noll index j) evaluated on the unit disk."""
    n, m = noll_to_zern(j)
    R = zernike_radial(n, m, rho)
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    ang = np.cos(m * theta) if m >= 0 else np.sin(abs(m) * theta)
    return norm * R * ang


def _zernike_basis_matrix(rho_support, theta_support, noll_indices):
    return np.stack([zernike_noll(j, rho_support, theta_support) for j in noll_indices], axis=1)


def fit_zernike_phase(phase, support, basis_matrix):
    """Least-squares project ``phase`` (restricted to ``support`` pixels) onto
    a precomputed Zernike basis matrix. Returns (coeffs, full-frame fitted
    phase with zeros outside ``support``)."""
    y = phase[support]
    coeffs, *_ = np.linalg.lstsq(basis_matrix, y, rcond=None)
    fitted = basis_matrix @ coeffs
    phase_fit = np.zeros_like(phase)
    phase_fit[support] = fitted
    return coeffs, phase_fit


# ---------------------------------------------------------------------------
# Optics: pupil support, angular-spectrum defocus propagation.
# ---------------------------------------------------------------------------

def _spatial_frequency_grid(ny, nx, xy_pixel_size_um):
    """Returns (fx, fy), each shape (ny, nx), in cycles/micron."""
    fx1d = np.fft.fftfreq(nx, d=xy_pixel_size_um)
    fy1d = np.fft.fftfreq(ny, d=xy_pixel_size_um)
    fy, fx = np.meshgrid(fy1d, fx1d, indexing="ij")
    return fx, fy


def pupil_support_mask(ny, nx, params: PSFModelParams):
    """Boolean mask of the pupil aperture (spatial frequencies within NA)."""
    fx, fy = _spatial_frequency_grid(ny, nx, params.xy_pixel_size_um)
    return (fx**2 + fy**2) <= (params.na / params.wavelength_um) ** 2


def _pupil_polar_grid(ny, nx, params: PSFModelParams):
    """Polar coordinates (rho, theta) normalized so rho=1 at the NA edge --
    the coordinate system Zernike polynomials are defined on."""
    fx, fy = _spatial_frequency_grid(ny, nx, params.xy_pixel_size_um)
    rho = np.sqrt(fx**2 + fy**2) / (params.na / params.wavelength_um)
    theta = np.arctan2(fy, fx)
    return rho, theta


def defocus_kernel_stack(ny, nx, z_positions_um, params: PSFModelParams, support):
    """Angular-spectrum defocus propagator exp(i*kz*z) for each z position,
    zeroed outside the pupil support (and wherever evanescent)."""
    fx, fy = _spatial_frequency_grid(ny, nx, params.xy_pixel_size_um)
    kz = 2 * np.pi * np.sqrt(
        np.clip((params.refractive_index / params.wavelength_um) ** 2 - fx**2 - fy**2, 0, None)
    )
    z_positions_um = np.asarray(z_positions_um)
    kernels = np.exp(1j * kz[None, :, :] * z_positions_um[:, None, None])
    return kernels * support[None, :, :]


def simulate_psf_from_pupil(pupil, z_positions_um, params: PSFModelParams):
    """Forward model: propagate a pupil function to each z position and
    return the resulting (Nz, Ny, Nx) intensity stack.

    ``np.fft.ifft2`` places the zero spatial position at index [0, 0], not
    the array's center, so the result is fftshift-ed to put the point
    source at the geometric center -- matching the convention
    ``extract_bead_psf`` uses when it centers a bead crop (and thus what any
    real measured PSF stack looks like). Skipping this shift is invisible in
    a self-consistent synthetic round-trip (ground truth and retrieval both
    silently agree on the same "corner-centered" convention) but produces a
    badly wrong fit against real, center-referenced data.
    """
    ny, nx = pupil.shape
    support = pupil_support_mask(ny, nx, params)
    kernels = defocus_kernel_stack(ny, nx, z_positions_um, params, support)
    stack = np.empty((len(z_positions_um), ny, nx), dtype=float)
    for zi in range(len(z_positions_um)):
        field = np.fft.fftshift(np.fft.ifft2(pupil * kernels[zi]))
        stack[zi] = np.abs(field) ** 2
    return stack


# ---------------------------------------------------------------------------
# Gerchberg-Saxton / Hanser phase retrieval.
# ---------------------------------------------------------------------------

def retrieve_pupil_phase(measured_psf, params: PSFModelParams, n_iterations=50,
                          zernike_regularize=True, noll_indices=DEFAULT_NOLL_INDICES):
    """Iteratively fit a pupil function to a measured through-focus PSF stack.

    Each iteration: propagate the current pupil estimate to every z-slice,
    replace the field's modulus with sqrt(measured intensity) at that slice
    (keeping the retrieved phase -- the modulus-constraint step), propagate
    back to the pupil plane, and average across slices for the next pupil
    phase estimate. If ``zernike_regularize``, that phase is projected onto a
    low-order Zernike basis each iteration, which both denoises the estimate
    and yields physically interpretable aberration coefficients.

    A nonzero recovered defocus term (Noll 4) is expected even for a
    well-focused system: the input PSF's Z-centering (from
    ``extract_bead_psf``) is an intensity-centroid convention, not a "true
    focal plane" convention, and the two are degenerate in this model.
    """
    nz, ny, nx = measured_psf.shape
    z_positions_um = (np.arange(nz) - nz // 2) * params.dz_um

    support = pupil_support_mask(ny, nx, params)
    kernels = defocus_kernel_stack(ny, nx, z_positions_um, params, support)
    amp_target = np.sqrt(np.clip(measured_psf, 0, None))

    rho, theta = _pupil_polar_grid(ny, nx, params)
    basis_matrix = _zernike_basis_matrix(rho[support], theta[support], noll_indices)

    pupil = support.astype(complex)
    residual_history = []

    for _ in range(n_iterations):
        accum = np.zeros((ny, nx), dtype=complex)
        residual_sq = 0.0
        for zi in range(nz):
            kernel = kernels[zi]
            # fftshift/ifftshift bracket the modulus-constraint step so it
            # operates in the same center-referenced frame as the measured
            # data (see simulate_psf_from_pupil for why this matters).
            field = np.fft.fftshift(np.fft.ifft2(pupil * kernel))
            mag = np.clip(np.abs(field), 1e-12, None)
            residual_sq += np.mean((mag - amp_target[zi]) ** 2)
            field_constrained = amp_target[zi] * (field / mag)
            accum += np.fft.fft2(np.fft.ifftshift(field_constrained)) * np.conj(kernel)
        residual_history.append(float(np.sqrt(residual_sq / nz)))

        phase = np.angle(accum / nz)
        if zernike_regularize:
            _coeffs, phase = fit_zernike_phase(phase, support, basis_matrix)
        pupil = support.astype(complex) * np.exp(1j * phase)

    final_coeffs, _ = fit_zernike_phase(np.angle(pupil), support, basis_matrix)
    return PhaseRetrievalResult(
        pupil=pupil,
        zernike_coeffs=dict(zip(noll_indices, (float(c) for c in final_coeffs))),
        noll_indices=tuple(noll_indices),
        residual_history=residual_history,
        params=params,
    )


def regenerate_and_save_psf(result: PhaseRetrievalResult, nz, out_tif_path, out_json_path=None,
                             target_energy=None):
    """Regenerate a smooth, noise-free PSF from the converged pupil (over an
    ``nz``-slice z-grid matching the input) and save it via the same
    ``tifffile.imwrite(..., imagej=True, metadata={"spacing": ...})`` call
    shape used by ``extract_bead_psf``, so it's a drop-in for ``psf_path=``.
    Optionally writes a JSON sidecar with the fitted parameters for
    reproducibility.
    """
    params = result.params
    z_positions_um = (np.arange(nz) - nz // 2) * params.dz_um
    psf = simulate_psf_from_pupil(result.pupil, z_positions_um, params)

    if target_energy is not None and psf.sum() > 0:
        psf = psf * (target_energy / psf.sum())

    tifffile.imwrite(
        out_tif_path,
        psf.astype(np.float32),
        imagej=True,
        metadata={"spacing": params.dz_um, "unit": "um"},
    )

    if out_json_path is not None:
        sidecar = {
            "na": params.na,
            "wavelength_um": params.wavelength_um,
            "refractive_index": params.refractive_index,
            "xy_pixel_size_um": params.xy_pixel_size_um,
            "dz_um": params.dz_um,
            "zernike_coeffs": {str(k): v for k, v in result.zernike_coeffs.items()},
            "residual_history": result.residual_history,
        }
        with open(out_json_path, "w") as f:
            json.dump(sidecar, f, indent=2)

    return psf


# ---------------------------------------------------------------------------
# I/O helpers -- deliberately local rather than importing opym's private
# _read_psf_dz across the bioimaging/opym_local boundary (psf_tools stays
# internal to bioimaging). Promote to a shared public helper if a third
# caller ever needs this.
# ---------------------------------------------------------------------------

def _read_dz_from_tif(path):
    with tifffile.TiffFile(str(path)) as tf:
        meta = tf.imagej_metadata or {}
        spacing = meta.get("spacing")
        return float(spacing) if spacing is not None else None


def default_wavelength_um_for_name(name):
    """Look up a default emission wavelength by matching ``name`` against
    analyze_channels' fluorophore patterns. Only used to pick which row of
    EMISSION_WAVELENGTH_UM applies -- never to parse a number out of the
    (excitation-labeled) pattern strings themselves."""
    from psf_tools.analyze_channels import parse_name_for_fluorophores

    for label in parse_name_for_fluorophores(name):
        if label in EMISSION_WAVELENGTH_UM:
            return EMISSION_WAVELENGTH_UM[label]
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fit a physical pupil-function model to a measured bead PSF via "
                    "iterative phase retrieval, and regenerate a smooth, noise-free "
                    "replacement PSF."
    )
    parser.add_argument("--input-psf", type=Path, required=True)
    parser.add_argument("--na", type=float, required=True, help="Detection objective NA.")
    wavelength_group = parser.add_mutually_exclusive_group(required=True)
    wavelength_group.add_argument("--wavelength-um", type=float,
                                   help="Emission wavelength of the calibration beads, in microns.")
    wavelength_group.add_argument("--fluorophore-name", type=str,
                                   help="Name to match against known fluorophores for a default "
                                        "emission wavelength (see EMISSION_WAVELENGTH_UM).")
    parser.add_argument("--n", dest="refractive_index", type=float, required=True,
                        help="Immersion/mounting medium refractive index.")
    parser.add_argument("--xy-pixel-size-um", type=float, default=0.136)
    parser.add_argument("--dz-um", type=float, default=None,
                        help="PSF z-step in microns. If omitted, read from the input PSF's "
                             "ImageJ 'spacing' tag.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--zernike-max-noll", type=int, default=11)
    parser.add_argument("--no-zernike-regularize", action="store_true")
    parser.add_argument("--output-psf", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    if args.wavelength_um is not None:
        wavelength_um = args.wavelength_um
    else:
        wavelength_um = default_wavelength_um_for_name(args.fluorophore_name)
        if wavelength_um is None:
            raise ValueError(
                f"Could not resolve an emission wavelength for "
                f"--fluorophore-name={args.fluorophore_name!r}. Pass --wavelength-um explicitly."
            )

    dz_um = args.dz_um
    if dz_um is None:
        dz_um = _read_dz_from_tif(args.input_psf)
        if dz_um is None:
            raise ValueError(
                f"dz_um could not be determined from '{args.input_psf}''s ImageJ 'spacing' tag. "
                "Pass --dz-um explicitly, or use a PSF saved by psf_tools.extract_bead_psf."
            )

    noll_indices = tuple(range(4, args.zernike_max_noll + 1))
    if not noll_indices:
        raise ValueError("--zernike-max-noll must be >= 4")

    measured_psf = tifffile.imread(args.input_psf).astype(float)
    while measured_psf.ndim > 3:
        measured_psf = measured_psf[0]

    params = PSFModelParams(
        na=args.na,
        wavelength_um=wavelength_um,
        refractive_index=args.refractive_index,
        xy_pixel_size_um=args.xy_pixel_size_um,
        dz_um=dz_um,
        zernike_noll_indices=noll_indices,
    )

    result = retrieve_pupil_phase(
        measured_psf, params,
        n_iterations=args.iterations,
        zernike_regularize=not args.no_zernike_regularize,
        noll_indices=noll_indices,
    )

    print("Fitted Zernike coefficients (rad):")
    for j, c in result.zernike_coeffs.items():
        n, m = noll_to_zern(j)
        print(f"  Noll {j} (n={n}, m={m}): {c:+.4f}")
    print(f"Final GS residual: {result.residual_history[-1]:.4f}")

    out_psf = args.output_psf or args.input_psf.with_name(args.input_psf.stem + "_phase_retrieved.tif")
    out_json = args.output_json or out_psf.with_suffix(".json")
    regenerate_and_save_psf(
        result, measured_psf.shape[0], out_psf, out_json,
        target_energy=float(measured_psf.sum()),
    )
    print(f"Saved phase-retrieved PSF to {out_psf} (dz_um={dz_um})")
    print(f"Saved fit parameters to {out_json}")


if __name__ == "__main__":
    main()
