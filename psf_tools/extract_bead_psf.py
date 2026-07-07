import csv
import typing
from pathlib import Path
import numpy as np
import napari
from magicgui import magic_factory
from scipy.ndimage import fourier_shift
from scipy.stats import trim_mean
import numpy.fft as fft
from skimage.feature import blob_log
from skimage.registration import phase_cross_correlation
import tifffile
import zarr
import dask.array as da
import warnings


@magic_factory(call_button="Load OME-TIFF")
def load_lazy_ome_tiff(file_path: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260209-YGBeads_PDMS_PSFmeasurement/1_60Ratio/0p1micron/0p1micron_MMStack_Pos0.ome.tif")) -> napari.types.LayerDataTuple:
    """Open an OME-TIFF as a lazy Dask array."""
    if file_path.is_dir():
        raise IsADirectoryError("Please select a specific TIFF file, not a directory.")

    store = tifffile.imread(file_path, aszarr=True)
    z = zarr.open(store, mode="r")
    lazy_data = da.from_zarr(z)
    return (lazy_data, {"name": file_path.name, "metadata": {"path": str(file_path)}}, "image")


@magic_factory(
    call_button="Detect Beads (LoG)",
    min_sigma={"step": 0.5},
    max_sigma={"step": 0.5},
    threshold={"step": 0.01},
)
def detect_beads(
    viewer: "napari.viewer.Viewer",
    image_layer: "napari.layers.Image",
    shapes_layer: typing.Optional["napari.layers.Shapes"] = None,
    min_sigma: float = 1.0,
    max_sigma: float = 3.0,
    num_sigma: int = 5,
    threshold: float = 0.1,
    overlap: float = 0.5,
    channel: int = 0,
):
    """
    Detect beads in 3D using Laplacian of Gaussian (blob_log).
    Note: For very large images, this might take a while as it computes in memory.
    """
    if image_layer is None:
        print("Please select an image layer.")
        return

    print("Computing image data... (this may take a moment for large files)")
    # Convert dask array to numpy array for processing if necessary
    data = np.asarray(image_layer.data)

    print(f"Running blob_log on data of shape {data.shape}...")

    # We want a 3D volume (Z, Y, X) for blob_log
    idx = None
    if data.ndim == 4:
        # Assuming (Z, C, Y, X) based on shape (200, 2, 2400, 2400)
        # User specified there are two channels.
        if data.shape[1] < data.shape[0]: # index 1 is likely channel
            process_data = data[:, channel, :, :]
            idx = (slice(None), channel, slice(None), slice(None))
            print(f"Data has 4 dims (assumed Z,C,Y,X). Extracted channel {channel}. Shape is now {process_data.shape}.")
        else: # maybe T, Z, Y, X
            process_data = data[0, :, :, :]
            idx = (0, slice(None), slice(None), slice(None))
            print(f"Data has 4 dims. Extracted T=0. Shape is now {process_data.shape}.")
    elif data.ndim > 4:
        # Just grab the first of everything else
        process_data = data[0, 0, :, :, :]
        idx = (0, 0, slice(None), slice(None), slice(None))
        print(f"Data has >4 dims. Extracted a 3D volume of shape {process_data.shape}.")
    else:
        process_data = data
        idx = ()

    y_offset = 0
    x_offset = 0

    if shapes_layer is not None and len(shapes_layer.data) > 0:
        shape_coords = shapes_layer.data[0]
        y_coords = shape_coords[:, -2]
        x_coords = shape_coords[:, -1]

        y_min = max(0, int(np.min(y_coords)))
        y_max = min(process_data.shape[-2], int(np.max(y_coords)))
        x_min = max(0, int(np.min(x_coords)))
        x_max = min(process_data.shape[-1], int(np.max(x_coords)))

        y_offset = y_min
        x_offset = x_min

        process_data = process_data[:, y_min:y_max, x_min:x_max]
        print(f"Cropped detection region to {process_data.shape} based on ROI.")

    # Normalize data to [0, 1] for consistent blob_log thresholding
    process_data = process_data.astype(np.float32)
    p_min = process_data.min()
    p_max = process_data.max()
    if p_max > p_min:
        process_data = (process_data - p_min) / (p_max - p_min)

    print(f"Normalized data for blob_log (min: {p_min}, max: {p_max}).")

    # blob_log works on 3D but can be slow
    blobs = blob_log(
        process_data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )

    if len(blobs) == 0:
        print("No beads detected with current parameters. Try lowering threshold.")
        return

    # blobs contains [z, y, x, sigma]
    coords = blobs[:, :3]

    # Apply spatial offset if ROI was used
    coords[:, 1] += y_offset
    coords[:, 2] += x_offset

    # Prepend the extra dimension indices if necessary
    if data.ndim == 4 and data.shape[1] < data.shape[0]:
        # Z, C, Y, X
        full_coords = np.zeros((coords.shape[0], data.ndim))
        full_coords[:, 0] = coords[:, 0]  # Z
        full_coords[:, 1] = channel       # C
        full_coords[:, 2] = coords[:, 1]  # Y
        full_coords[:, 3] = coords[:, 2]  # X
        coords = full_coords
    elif data.ndim == 4:
        # T, Z, Y, X
        full_coords = np.zeros((coords.shape[0], data.ndim))
        full_coords[:, 0] = 0             # T
        full_coords[:, 1] = coords[:, 0]  # Z
        full_coords[:, 2] = coords[:, 1]  # Y
        full_coords[:, 3] = coords[:, 2]  # X
        coords = full_coords
    elif data.ndim > 3:
        # general fallback
        full_coords = np.zeros((coords.shape[0], data.ndim))
        full_coords[:, -3:] = coords
        coords = full_coords

    print(f"Detected {len(coords)} beads.")

    # Update or add points layer
    if "Beads" in viewer.layers:
        viewer.layers["Beads"].data = coords
    else:
        viewer.add_points(
            coords,
            name="Beads",
            size=5,
            face_color="transparent",
            border_color="red",
            ndim=data.ndim
        )


@magic_factory(call_button="Save Points")
def save_points(points_layer: "napari.layers.Points", save_path: Path = Path("./bead_coords.csv")):
    """Save the current points layer coordinates to a CSV file."""
    if points_layer is not None and len(points_layer.data) > 0:
        np.savetxt(save_path, points_layer.data, delimiter=",")
        print(f"Saved {len(points_layer.data)} points to {save_path.absolute()}")
    else:
        print("No points to save.")

@magic_factory(call_button="Load Points")
def load_points(viewer: "napari.viewer.Viewer", load_path: Path = Path("./bead_coords.csv")):
    """Load points coordinates from a CSV file into the Beads layer."""
    if load_path.exists():
        coords = np.loadtxt(load_path, delimiter=",")
        if coords.ndim == 1:
            coords = coords[np.newaxis, :]

        if "Beads" in viewer.layers:
            viewer.layers["Beads"].data = coords
        else:
            viewer.add_points(
                coords,
                name="Beads",
                size=5,
                face_color="transparent",
                border_color="red",
                ndim=coords.shape[1]
            )
        print(f"Loaded {len(coords)} points from {load_path.absolute()}")
    else:
        print(f"File not found: {load_path.absolute()}")


# ---------------------------------------------------------------------------
# Reusable, napari-independent helpers for bead-crop extraction, centering,
# quality scoring, and averaging. Kept importable (no napari/viewer args) so
# they can be exercised directly from tests and from other scripts (e.g. a
# future phase-retrieval PSF fitter) without spinning up a viewer.
# ---------------------------------------------------------------------------

def _fourier_shift_3d(volume, shift_vec):
    """Exact sub-pixel shift of a 3D volume via the Fourier shift theorem."""
    vol_fft = fft.fftn(volume)
    shifted_fft = fourier_shift(vol_fft, shift_vec)
    return fft.ifftn(shifted_fft).real


def _mean_shift_center(crop_sub, n_iter=10, window_radius=5.0):
    """Iterative weighted mean-shift to find a robust sub-pixel bead center.

    Starts at the geometric center of the crop (never argmax -- that would
    blindly jump to any brighter neighboring bead/dirt in the crop window)
    and repeatedly recenters a small window on the local intensity centroid.

    Returns (z0, y0, x0, weight, residual_px). ``weight`` is 0 if the window
    ever went fully dark (empty/zero crop). ``residual_px`` is the distance
    moved on the final update -- a bead whose center is still moving a lot
    on the last iteration hasn't converged, usually because a neighboring
    blob or debris is pulling on the centroid.
    """
    Z, Y, X = crop_sub.shape
    z0, y0, x0 = Z / 2.0 - 0.5, Y / 2.0 - 0.5, X / 2.0 - 0.5
    z_grid, y_grid, x_grid = np.ogrid[0:Z, 0:Y, 0:X]

    weight = 0.0
    residual_px = 0.0
    for _ in range(n_iter):
        window_mask = (
            (z_grid - z0) ** 2 / window_radius**2
            + (y_grid - y0) ** 2 / window_radius**2
            + (x_grid - x0) ** 2 / window_radius**2
        ) <= 1
        windowed = crop_sub * window_mask
        weight = windowed.sum()
        if weight == 0:
            break
        z0_new = (z_grid * windowed).sum() / weight
        y0_new = (y_grid * windowed).sum() / weight
        x0_new = (x_grid * windowed).sum() / weight
        residual_px = float(np.sqrt((z0_new - z0) ** 2 + (y0_new - y0) ** 2 + (x0_new - x0) ** 2))
        z0, y0, x0 = z0_new, y0_new, x0_new

    return z0, y0, x0, weight, residual_px


def _windowed_covariance(crop_sub, center, radius_z, radius_xy):
    """3D covariance matrix of the intensity distribution around ``center``."""
    z0, y0, x0 = center
    Z, Y, X = crop_sub.shape
    z_grid, y_grid, x_grid = np.ogrid[0:Z, 0:Y, 0:X]

    cov_mask = (
        (z_grid - z0) ** 2 / (radius_z * 0.5) ** 2
        + (y_grid - y0) ** 2 / (radius_xy * 0.5) ** 2
        + (x_grid - x0) ** 2 / (radius_xy * 0.5) ** 2
    ) <= 1
    windowed_cov = crop_sub * cov_mask
    weight_cov = windowed_cov.sum()

    if weight_cov <= 0:
        return np.eye(3)

    dz = z_grid - z0
    dy = y_grid - y0
    dx = x_grid - x0
    c_zz = (dz**2 * windowed_cov).sum() / weight_cov
    c_yy = (dy**2 * windowed_cov).sum() / weight_cov
    c_xx = (dx**2 * windowed_cov).sum() / weight_cov
    c_zy = (dz * dy * windowed_cov).sum() / weight_cov
    c_zx = (dz * dx * windowed_cov).sum() / weight_cov
    c_yx = (dy * dx * windowed_cov).sum() / weight_cov
    return np.array([[c_zz, c_zy, c_zx], [c_zy, c_yy, c_yx], [c_zx, c_yx, c_xx]])


def _soft_ellipsoid_mask(shape, center, cov_matrix, inner_sigma=3.0, outer_sigma=5.0):
    """Soft ellipsoid mask: 1.0 inside ``inner_sigma``, rolls smoothly to 0.0
    at ``outer_sigma``. Prevents sharp-cutoff ringing artifacts during
    Fourier-shift centering, and (at wider sigmas) suppresses the residual
    noise floor in an averaged PSF's tails.
    """
    Z, Y, X = shape
    z0, y0, x0 = center
    z_grid, y_grid, x_grid = np.ogrid[0:Z, 0:Y, 0:X]

    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(3)

    dz = z_grid - z0
    dy = y_grid - y0
    dx = x_grid - x0
    mahal_sq = (
        inv_cov[0, 0] * dz**2 + inv_cov[1, 1] * dy**2 + inv_cov[2, 2] * dx**2
        + 2 * inv_cov[0, 1] * dz * dy + 2 * inv_cov[0, 2] * dz * dx + 2 * inv_cov[1, 2] * dy * dx
    )
    mahal = np.sqrt(np.clip(mahal_sq, 0, None))
    return np.clip((outer_sigma - mahal) / (outer_sigma - inner_sigma), 0, 1)


def _extract_crop(data, pt, radius_z, radius_xy):
    """Crop a (2*radius_z+1, 2*radius_xy+1, 2*radius_xy+1) box around a bead
    point, handling (Z,C,Y,X) / (T,Z,Y,X) / general (..., Z, Y, X) layouts
    the same way the original single-purpose loop did. Returns None (and
    prints why) if the box would run off the edge of the volume.
    """
    shape = data.shape
    pt_r = np.round(pt).astype(int)

    if data.ndim == 4 and data.shape[1] < data.shape[0]:
        # Z, C, Y, X
        z, c, y, x = pt_r
        z_min, z_max = z - radius_z, z + radius_z + 1
        y_min, y_max = y - radius_xy, y + radius_xy + 1
        x_min, x_max = x - radius_xy, x + radius_xy + 1
        if (z_min < 0 or z_max > shape[0] or
                y_min < 0 or y_max > shape[2] or
                x_min < 0 or x_max > shape[3]):
            return None
        return data[(slice(z_min, z_max), c, slice(y_min, y_max), slice(x_min, x_max))]

    # General (..., Z, Y, X), including (T, Z, Y, X)
    idx_tuple = tuple(pt_r[:-3])
    z, y, x = pt_r[-3:]
    z_min, z_max = z - radius_z, z + radius_z + 1
    y_min, y_max = y - radius_xy, y + radius_xy + 1
    x_min, x_max = x - radius_xy, x + radius_xy + 1
    if (z_min < 0 or z_max > shape[-3] or
            y_min < 0 or y_max > shape[-2] or
            x_min < 0 or x_max > shape[-1]):
        return None
    return data[idx_tuple + (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))]


def _isolation_score(points, i, radius_z, radius_xy):
    """Radius-normalized distance from bead ``i`` to its nearest *other*
    detected point (anisotropic: Z distances scaled by ``radius_z``, XY by
    ``radius_xy``). A value of 1.0 means the nearest neighbor sits right at
    the edge of this bead's own crop box -- i.e. the crops likely overlap.
    Computed over the *full* input point set (not just accepted beads),
    since a dim rejected blob can still optically contaminate a nearby crop.
    """
    pt = np.asarray(points[i])[-3:]
    min_dist = np.inf
    for j, other in enumerate(points):
        if j == i:
            continue
        o = np.asarray(other)[-3:]
        dz = (pt[0] - o[0]) / radius_z
        dy = (pt[1] - o[1]) / radius_xy
        dx = (pt[2] - o[2]) / radius_xy
        dist = float(np.sqrt(dz**2 + dy**2 + dx**2))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _snr_score(crop_sub, soft_mask):
    """Peak intensity (inside the bead mask) vs. background noise std
    (outside the mask, within the same crop box)."""
    signal_region = soft_mask > 0.5
    background_region = soft_mask < 0.05
    peak = float(crop_sub[signal_region].max()) if signal_region.any() else 0.0
    bg_std = float(crop_sub[background_region].std()) if background_region.any() else 0.0
    if bg_std <= 0:
        return np.inf if peak > 0 else 0.0
    return peak / bg_std


def _mirror_symmetry_score(final_crop):
    """Lateral (XY) 180-degree self-symmetry of the crop's brightest z-slice.

    Deliberately XY-only: real aberrated PSFs are legitimately asymmetric
    top-to-bottom along Z (that asymmetry is exactly the aberration signal a
    downstream phase-retrieval fit would want to recover), so scoring Z
    symmetry would systematically reject the most informative beads. Only
    genuinely lateral defects (overlapping neighbor, comet-tail dirt) show up
    as low XY self-correlation.
    """
    slice_sums = final_crop.sum(axis=(1, 2))
    z_peak = int(np.argmax(slice_sums))
    img = final_crop[z_peak]
    mirrored = img[::-1, ::-1]
    if img.std() == 0 or mirrored.std() == 0:
        return 0.0
    return float(np.corrcoef(img.ravel(), mirrored.ravel())[0, 1])


def score_bead_crops(data, points, radius_z, radius_xy, normalize=True):
    """Extract, center, and quality-score a crop around every detected bead.

    Reproduces the original (pre-refactor) per-bead pipeline exactly --
    background subtraction, iterative mean-shift centering, covariance-based
    soft ellipsoid masking, and exact Fourier-shift sub-pixel centering --
    then attaches quality scores used for automatic accept/reject gating.

    Returns a list of dicts (one per bead that survived extraction), each
    with keys: index, point, final_crop, isolation, snr, mirror_symmetry,
    meanshift_residual_px, energy, accepted (defaults to True; call
    ``apply_quality_gates`` to actually gate on the scores).
    """
    scores = []
    for i, pt in enumerate(points):
        crop = _extract_crop(data, pt, radius_z, radius_xy)
        if crop is None:
            print(f"Point {i} at {np.round(pt).astype(int)} is too close to edge, skipping.")
            continue

        crop_float = crop.astype(float)
        bg = np.percentile(crop_float, 10)
        crop_sub = np.clip(crop_float - bg, 0, None)

        z0, y0, x0, weight, residual_px = _mean_shift_center(crop_sub)
        if weight == 0:
            print(f"Point {i} crop is empty or all zeros, skipping.")
            continue

        cov_matrix = _windowed_covariance(crop_sub, (z0, y0, x0), radius_z, radius_xy)
        soft_mask = _soft_ellipsoid_mask(crop_sub.shape, (z0, y0, x0), cov_matrix)

        masked_crop = crop_sub * soft_mask
        Z, Y, X = crop_sub.shape
        center_target = np.array([Z / 2 - 0.5, Y / 2 - 0.5, X / 2 - 0.5])
        shift_vec = center_target - np.array([z0, y0, x0])
        final_crop = _fourier_shift_3d(masked_crop, shift_vec)
        final_crop[final_crop < 0] = 0

        energy = float(np.sum(final_crop))
        if normalize:
            if energy > 0:
                final_crop = final_crop / energy
            else:
                print(f"Point {i} has 0 energy, skipping.")
                continue

        scores.append({
            "index": i,
            "point": np.round(pt).astype(int),
            "final_crop": final_crop,
            "isolation": _isolation_score(points, i, radius_z, radius_xy),
            "snr": _snr_score(crop_sub, soft_mask),
            "mirror_symmetry": _mirror_symmetry_score(final_crop),
            "meanshift_residual_px": residual_px,
            "energy": energy,
            "accepted": True,
        })

    return scores


def apply_quality_gates(
    scores,
    min_isolation_radii=1.5,
    min_snr=5.0,
    min_mirror_symmetry=0.5,
    max_meanshift_residual_px=0.1,
):
    """Threshold already-computed per-bead scores, setting ``accepted`` in
    place. Kept separate from ``score_bead_crops`` so thresholds can be
    re-tuned without re-running the (slower) extraction/centering pass.
    """
    for s in scores:
        s["accepted"] = (
            s["isolation"] >= min_isolation_radii
            and s["snr"] >= min_snr
            and s["mirror_symmetry"] >= min_mirror_symmetry
            and s["meanshift_residual_px"] <= max_meanshift_residual_px
        )
    return scores


def bootstrap_reregister(crops, upsample_factor=20):
    """Re-register each bead crop against a provisional median template via
    subpixel cross-correlation, instead of trusting only the intensity
    mean-shift centroid (which is biased by any residual asymmetric
    background or partial neighboring blob inside the crop window).
    """
    if len(crops) < 2:
        return crops

    template = np.median(np.stack(crops, axis=0), axis=0)
    registered = []
    for crop in crops:
        shift_est, _error, _phasediff = phase_cross_correlation(
            template, crop, upsample_factor=upsample_factor, normalization=None
        )
        registered.append(_fourier_shift_3d(crop, shift_est))
    return registered


def robust_average(crops, combine="trimmed_mean", trim_fraction=0.1):
    """Combine bead crops into a single averaged PSF. ``combine="mean"``
    reproduces the original plain-average behavior exactly."""
    stack = np.stack(crops, axis=0)
    if combine == "mean":
        return stack.mean(axis=0)
    if combine == "median":
        return np.median(stack, axis=0)
    if combine == "trimmed_mean":
        return trim_mean(stack, trim_fraction, axis=0)
    raise ValueError(f"Unknown combine method: {combine!r}")


def export_bead_scores_csv(scores, csv_path):
    """Write one row per scored bead so automatic gating can be sanity
    checked against a prior manually-curated coordinate CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "z", "y", "x", "isolation", "snr", "mirror_symmetry",
            "meanshift_residual_px", "energy", "accepted",
        ])
        for s in scores:
            z, y, x = s["point"][-3:]
            writer.writerow([
                s["index"], z, y, x, s["isolation"], s["snr"],
                s["mirror_symmetry"], s["meanshift_residual_px"], s["energy"],
                s["accepted"],
            ])


@magic_factory(
    call_button="Extract & Average PSF",
    radius_z={"min": 1, "max": 100},
    radius_xy={"min": 1, "max": 100},
    combine={"choices": ["trimmed_mean", "median", "mean"]},
)
def extract_average_psf(
    viewer: "napari.viewer.Viewer",
    image_layer: "napari.layers.Image",
    points_layer: "napari.layers.Points",
    radius_z: int = 40,
    radius_xy: int = 15,
    normalize: bool = True,
    rotate_90: bool = True,
    dz_psf: float = 0.1,
    auto_reject: bool = True,
    min_isolation_radii: float = 1.5,
    min_snr: float = 5.0,
    min_mirror_symmetry: float = 0.5,
    max_meanshift_residual_px: float = 0.1,
    bootstrap_reregister_beads: bool = True,
    bootstrap_upsample_factor: int = 20,
    combine: str = "trimmed_mean",
    trim_fraction: float = 0.1,
    apodize_tails: bool = True,
    apodize_inner_sigma_mult: float = 6.0,
    apodize_outer_sigma_mult: float = 9.0,
    scores_csv_path: Path = Path("./bead_scores.csv"),
    save_path: Path = Path("./averaged_psf.tif"),
):
    """
    Extracts crops around each point, aligns them by center of mass,
    normalizes energy, and averages them to create a master PSF.

    Automatic quality gating (``auto_reject``) replaces manually curating
    "keeper" point CSVs across repeated napari sessions: each bead is scored
    for isolation, SNR, mirror symmetry, and mean-shift convergence, and
    beads that fail are dropped automatically (see "Inspect Bead Scores" to
    tune thresholds without re-extracting). Set
    ``auto_reject=False, bootstrap_reregister_beads=False, combine="mean",
    apodize_tails=False`` to reproduce the original plain-average behavior
    exactly.

    dz_psf is the z-step (in microns) the PSF bead stack was acquired at.
    It is saved into the output tiff's metadata (ImageJ 'spacing' tag) so
    downstream deconvolution can resample the PSF to match a given
    dataset's actual z-step instead of silently assuming they match.
    """
    if image_layer is None or points_layer is None:
        print("Please provide both an Image and Points layer.")
        return

    points = points_layer.data
    if len(points) == 0:
        print("No points found in Points layer.")
        return

    data = np.asarray(image_layer.data)

    if data.ndim < 3:
        print("Image must be at least 3D (Z, Y, X).")
        return

    scores = score_bead_crops(data, points, radius_z, radius_xy, normalize=normalize)
    if not scores:
        print("No valid crops could be extracted.")
        return

    if auto_reject:
        apply_quality_gates(
            scores,
            min_isolation_radii=min_isolation_radii,
            min_snr=min_snr,
            min_mirror_symmetry=min_mirror_symmetry,
            max_meanshift_residual_px=max_meanshift_residual_px,
        )

    export_bead_scores_csv(scores, scores_csv_path)
    print(f"Wrote per-bead quality scores to {Path(scores_csv_path).absolute()}")

    accepted_crops = [s["final_crop"] for s in scores if s["accepted"]]
    print(f"Accepted {len(accepted_crops)}/{len(scores)} beads.")
    if not accepted_crops:
        print("No beads passed quality gates. Loosen thresholds or disable auto_reject.")
        return

    if bootstrap_reregister_beads:
        accepted_crops = bootstrap_reregister(accepted_crops, upsample_factor=bootstrap_upsample_factor)

    avg_psf = robust_average(accepted_crops, combine=combine, trim_fraction=trim_fraction)

    if apodize_tails:
        Z, Y, X = avg_psf.shape
        center_final = (Z / 2 - 0.5, Y / 2 - 0.5, X / 2 - 0.5)
        cov_final = _windowed_covariance(avg_psf, center_final, radius_z, radius_xy)
        tail_mask = _soft_ellipsoid_mask(
            avg_psf.shape, center_final, cov_final,
            inner_sigma=apodize_inner_sigma_mult, outer_sigma=apodize_outer_sigma_mult,
        )
        avg_psf = avg_psf * tail_mask

    if rotate_90:
        print("Rotating PSF 90 degrees counter-clockwise in XY plane...")
        avg_psf = np.rot90(avg_psf, k=1, axes=(1, 2))

    # Save the result
    save_path = Path(save_path)
    if save_path.parent.exists() or save_path.parent == Path(""):
        tifffile.imwrite(
            save_path,
            avg_psf.astype(np.float32),
            imagej=True,
            metadata={"spacing": dz_psf, "unit": "um"},
        )
        print(f"Saved averaged PSF to {save_path.absolute()} (dz_psf={dz_psf} um)")
    else:
        print(f"Warning: Directory {save_path.parent} does not exist. Not saving.")

    # Display the result
    if "Averaged PSF" in viewer.layers:
        viewer.layers["Averaged PSF"].data = avg_psf
    else:
        viewer.add_image(avg_psf, name="Averaged PSF", colormap="viridis")

    print("Done!")


@magic_factory(
    call_button="Inspect Bead Scores",
    radius_z={"min": 1, "max": 100},
    radius_xy={"min": 1, "max": 100},
)
def inspect_bead_scores(
    image_layer: "napari.layers.Image",
    points_layer: "napari.layers.Points",
    radius_z: int = 40,
    radius_xy: int = 15,
    min_isolation_radii: float = 1.5,
    min_snr: float = 5.0,
    min_mirror_symmetry: float = 0.5,
    max_meanshift_residual_px: float = 0.1,
    scores_csv_path: Path = Path("./bead_scores.csv"),
):
    """Score bead crops and export accept/reject decisions without
    averaging or saving a PSF -- for quickly tuning quality-gate thresholds
    before committing to a full "Extract & Average PSF" run."""
    if image_layer is None or points_layer is None:
        print("Please provide both an Image and Points layer.")
        return

    points = points_layer.data
    if len(points) == 0:
        print("No points found in Points layer.")
        return

    data = np.asarray(image_layer.data)
    if data.ndim < 3:
        print("Image must be at least 3D (Z, Y, X).")
        return

    scores = score_bead_crops(data, points, radius_z, radius_xy)
    apply_quality_gates(
        scores,
        min_isolation_radii=min_isolation_radii,
        min_snr=min_snr,
        min_mirror_symmetry=min_mirror_symmetry,
        max_meanshift_residual_px=max_meanshift_residual_px,
    )
    export_bead_scores_csv(scores, scores_csv_path)
    n_accepted = sum(1 for s in scores if s["accepted"])
    print(f"Scored {len(scores)} beads, {n_accepted} pass current thresholds. "
          f"Wrote {Path(scores_csv_path).absolute()}")


if __name__ == "__main__":
    viewer = napari.Viewer()

    viewer.add_shapes(name="Detection ROI", edge_color="red", face_color="transparent", edge_width=5)
    # Also add an empty Points layer so it's ready for manual clicking
    viewer.add_points(np.empty((0, 4)), name="Beads", size=5, face_color="transparent", border_color="red", ndim=4)

    viewer.window.add_dock_widget(load_lazy_ome_tiff(), name="1. Load OME-TIFF", area="right")
    viewer.window.add_dock_widget(detect_beads(), name="2. Detect Beads", area="right")

    # Save/Load widgets grouped together
    from qtpy.QtWidgets import QWidget, QVBoxLayout
    io_widget = QWidget()
    io_layout = QVBoxLayout()
    io_layout.addWidget(save_points().native)
    io_layout.addWidget(load_points().native)
    io_widget.setLayout(io_layout)
    viewer.window.add_dock_widget(io_widget, name="2.5 Save/Load Points", area="right")

    viewer.window.add_dock_widget(extract_average_psf(), name="3. Extract & Average PSF", area="right")
    viewer.window.add_dock_widget(inspect_bead_scores(), name="3.5 Inspect Bead Scores", area="right")

    napari.run()
