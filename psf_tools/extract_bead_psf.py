import typing
from pathlib import Path
import numpy as np
import napari
from magicgui import magic_factory
from scipy.ndimage import fourier_shift
import numpy.fft as fft
from skimage.feature import blob_log
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


@magic_factory(
    call_button="Extract & Average PSF",
    radius_z={"min": 1, "max": 100},
    radius_xy={"min": 1, "max": 100},
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
    save_path: Path = Path("./averaged_psf.tif"),
):
    """
    Extracts crops around each point, aligns them by center of mass,
    normalizes energy, and averages them to create a master PSF.

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
    
    valid_crops = []
    
    shape = data.shape
    ndim = data.ndim
    
    # We assume the last 3 dims are Z, Y, X.
    if ndim < 3:
        print("Image must be at least 3D (Z, Y, X).")
        return

    for i, pt in enumerate(points):
        pt = np.round(pt).astype(int)
        
        # Determine the bounding box for the crop
        # Handle cases with T/C dimensions:
        if data.ndim == 4 and data.shape[1] < data.shape[0]:
            # Z, C, Y, X
            z = pt[0]
            c = pt[1]
            y = pt[2]
            x = pt[3]
            
            z_min, z_max = z - radius_z, z + radius_z + 1
            y_min, y_max = y - radius_xy, y + radius_xy + 1
            x_min, x_max = x - radius_xy, x + radius_xy + 1
            
            # Check bounds
            if (z_min < 0 or z_max > shape[0] or 
                y_min < 0 or y_max > shape[2] or 
                x_min < 0 or x_max > shape[3]):
                print(f"Point {i} at {pt} is too close to edge, skipping.")
                continue
                
            crop = data[(slice(z_min, z_max), c, slice(y_min, y_max), slice(x_min, x_max))]
        else:
            # Assume general (..., Z, Y, X)
            idx_tuple = tuple(pt[:-3])
            z, y, x = pt[-3:]
            
            z_min, z_max = z - radius_z, z + radius_z + 1
            y_min, y_max = y - radius_xy, y + radius_xy + 1
            x_min, x_max = x - radius_xy, x + radius_xy + 1
            
            # Check bounds
            if (z_min < 0 or z_max > shape[-3] or 
                y_min < 0 or y_max > shape[-2] or 
                x_min < 0 or x_max > shape[-1]):
                print(f"Point {i} at {pt} is too close to edge, skipping.")
                continue
                
            crop = data[idx_tuple + (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))]
        
        crop_float = crop.astype(float)
        bg = np.percentile(crop_float, 10)
        crop_sub = np.clip(crop_float - bg, 0, None)
        
        # 1. Iterative Mean Shift to find robust sub-pixel center
        Z, Y, X = crop_sub.shape
        # Initialize at the geometric center (the exact coordinate the user clicked!)
        # Do NOT use argmax here, otherwise it will blindly jump to any brighter neighboring bead/dirt in the large crop window!
        z0, y0, x0 = Z / 2.0 - 0.5, Y / 2.0 - 0.5, X / 2.0 - 0.5
        z_grid, y_grid, x_grid = np.ogrid[0:Z, 0:Y, 0:X]
        
        for _ in range(10):
            # 5 pixel radius window around current peak perfectly isolates bead
            window_mask = ((z_grid - z0)**2 / 5**2 + (y_grid - y0)**2 / 5**2 + (x_grid - x0)**2 / 5**2) <= 1
            windowed = crop_sub * window_mask
            weight = windowed.sum()
            if weight == 0: break
            z0 = (z_grid * windowed).sum() / weight
            y0 = (y_grid * windowed).sum() / weight
            x0 = (x_grid * windowed).sum() / weight
            
        if weight == 0:
            print(f"Point {i} crop is empty or all zeros, skipping.")
            continue
            
        # 2. Compute 3D Covariance Matrix of the bead
        cov_mask = ((z_grid - z0)**2 / (radius_z*0.5)**2 + (y_grid - y0)**2 / (radius_xy*0.5)**2 + (x_grid - x0)**2 / (radius_xy*0.5)**2) <= 1
        windowed_cov = crop_sub * cov_mask
        weight_cov = windowed_cov.sum()
        
        if weight_cov > 0:
            dz = z_grid - z0
            dy = y_grid - y0
            dx = x_grid - x0
            c_zz = (dz**2 * windowed_cov).sum() / weight_cov
            c_yy = (dy**2 * windowed_cov).sum() / weight_cov
            c_xx = (dx**2 * windowed_cov).sum() / weight_cov
            c_zy = (dz * dy * windowed_cov).sum() / weight_cov
            c_zx = (dz * dx * windowed_cov).sum() / weight_cov
            c_yx = (dy * dx * windowed_cov).sum() / weight_cov
            cov_matrix = np.array([[c_zz, c_zy, c_zx], [c_zy, c_yy, c_yx], [c_zx, c_yx, c_xx]])
        else:
            cov_matrix = np.eye(3)
            
        # 3. Create a Soft Ellipsoid Mask from covariance
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except:
            inv_cov = np.eye(3)
            
        dz = z_grid - z0
        dy = y_grid - y0
        dx = x_grid - x0
        mahal_sq = (
            inv_cov[0,0]*dz**2 + inv_cov[1,1]*dy**2 + inv_cov[2,2]*dx**2 +
            2*inv_cov[0,1]*dz*dy + 2*inv_cov[0,2]*dz*dx + 2*inv_cov[1,2]*dy*dx
        )
        
        mahal = np.sqrt(np.clip(mahal_sq, 0, None))
        # Soft mask: 1.0 inside 3 sigma, smoothly rolls to 0.0 at 5 sigma
        # This prevents sharp cutoff artifacts (ringing/crosses) during Fourier shift
        soft_mask = np.clip((5.0 - mahal) / 2.0, 0, 1)
        
        # 4. Mask and Sub-pixel shift
        masked_crop = crop_sub * soft_mask
        
        center_target = np.array([Z/2 - 0.5, Y/2 - 0.5, X/2 - 0.5])
        shift_vec = center_target - np.array([z0, y0, x0])
        
        # Align perfectly to center pixel using mathematically exact Fourier shift
        img_fft = fft.fftn(masked_crop)
        img_shifted_fft = fourier_shift(img_fft, shift_vec)
        final_crop = fft.ifftn(img_shifted_fft).real
        
        # Ensure no negative ringing artifacts
        final_crop[final_crop < 0] = 0
        
        if normalize:
            # Normalize energy
            energy = np.sum(final_crop)
            if energy > 0:
                final_crop /= energy
            else:
                print(f"Point {i} has 0 energy, skipping.")
                continue
                
        valid_crops.append(final_crop)

    if not valid_crops:
        print("No valid crops could be extracted.")
        return
        
    print(f"Averaging {len(valid_crops)} valid crops...")
    psf_stack = np.stack(valid_crops, axis=0)
    avg_psf = np.mean(psf_stack, axis=0)
    
    if rotate_90:
        print("Rotating PSF 90 degrees counter-clockwise in XY plane...")
        avg_psf = np.rot90(avg_psf, k=1, axes=(1, 2))

    # Save the result
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
    
    napari.run()
