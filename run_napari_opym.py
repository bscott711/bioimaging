import sys
import time
import napari
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from pathlib import Path
from xml.etree import ElementTree as ET
import tifffile
import zarr
import dask.array as da

from opym.petakit import (
    submit_remote_crop_job,
    submit_remote_deskew_job,
    submit_remote_decon_job,
    wait_for_job,
)


def _get_z_step_um(path: Path) -> float | None:
    """Extract PhysicalSizeZ (in µm) from OME-TIFF metadata."""
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.ome_metadata:
                root = ET.fromstring(tif.ome_metadata)
                # OME-XML namespaces vary; iterating avoids strict XPath dependency
                for pixels in root.iter():
                    if pixels.tag.endswith("Pixels"):
                        z_step = pixels.get("PhysicalSizeZ")
                        unit = pixels.get("PhysicalSizeZUnit", "µm")
                        if z_step and unit in ("µm", "um", "micron", "micrometer"):
                            return float(z_step)
    except Exception:
        pass  # Silently fall back to manual input
    return None


@thread_worker
def _run_job_chain(
    crop_job_path: Path, output_dir: Path, z_step_um: float, rotate_90: bool,
    sheet_angle_deg: float = 30.0,
    objective_scan: bool = True, z_stage_scan: bool = False, reverse: bool = False,
    run_decon: bool = True,
    psf_path: Path | str | None = None,
    decon_iters: int = 10,
    gpu_decon: bool = False,
):
    """Waits for jobs to finish with a live console timer, then chains them."""
    job_name = crop_job_path.name
    base_dir = crop_job_path.parent.parent  # Resolves to ~/petakit_jobs
    done_path = base_dir / "completed" / job_name
    fail_path = base_dir / "failed" / job_name

    print(f"\n🚀 Crop job submitted to HPC queue: {job_name}")
    
    start_time = time.time()
    success = False
    
    # 1. Custom live-updating polling loop for the Crop Job
    while True:
        elapsed = int(time.time() - start_time)
        
        if done_path.exists():
            sys.stdout.write(f"\r✅ Crop job completed in {elapsed}s!{' ' * 20}\n")
            sys.stdout.flush()
            success = True
            break
        elif fail_path.exists():
            sys.stdout.write(f"\r❌ Crop job failed after {elapsed}s!{' ' * 20}\n")
            sys.stdout.flush()
            success = False
            break
        
        # Live ticking timer that overwrites its own line
        sys.stdout.write(f"\r⏱️  Crop Elapsed: {elapsed}s | Status: MATLAB is processing...{' ' * 5}")
        sys.stdout.flush()
        time.sleep(2)

    if success:
        deskew_input = output_dir

        if run_decon and psf_path:
            print("\n🚀 Submitting standalone Decon job to PetaKit...")
            decon_ticket = submit_remote_decon_job(
                input_target=output_dir,
                psf_paths=psf_path,
                iterations=decon_iters,
                gpu_job=gpu_decon,
                skewed=True,
                result_dir_name="Decon",
                channel_patterns=[output_dir.name],
            )
            
            d_name = decon_ticket.name
            d_done = base_dir / "completed" / d_name
            d_fail = base_dir / "failed" / d_name
            
            d_start = time.time()
            decon_success = False
            while True:
                d_elapsed = int(time.time() - d_start)
                if d_done.exists():
                    sys.stdout.write(f"\r✅ Decon job completed in {d_elapsed}s!{' ' * 20}\n")
                    sys.stdout.flush()
                    decon_success = True
                    break
                elif d_fail.exists():
                    sys.stdout.write(f"\r❌ Decon job failed after {d_elapsed}s!{' ' * 20}\n")
                    sys.stdout.flush()
                    break
                
                sys.stdout.write(f"\r⏱️  Decon Elapsed: {d_elapsed}s | Status: GPUs are crunching...{' ' * 5}")
                sys.stdout.flush()
                time.sleep(2)

            if not decon_success:
                print("\n❌ Decon job failed. Aborting deskew submission.")
                return
            
            deskew_input = output_dir / "Decon"

        print("\n🚀 Submitting Deskew job to PetaKit...")
        deskew_ticket = submit_remote_deskew_job(
            input_target=deskew_input,
            z_step_um=z_step_um,
            sheet_angle_deg=sheet_angle_deg,
            deskew=True,
            rotate=True,
            objective_scan=objective_scan,
            z_stage_scan=z_stage_scan,
            reverse=reverse,
            gpu_decon=gpu_decon,
            crop_was_rotated=rotate_90,
            psf_path=None,
            n_iters=None,
            channel_patterns=[output_dir.name],
        )
        print(f"✅ Deskew job successfully queued: {deskew_ticket.name}")
        
        # 2. Add a live timer for the Deskew Job too!
        d_name = deskew_ticket.name
        d_done = base_dir / "completed" / d_name
        d_fail = base_dir / "failed" / d_name
        
        d_start = time.time()
        while True:
            d_elapsed = int(time.time() - d_start)
            if d_done.exists():
                sys.stdout.write(f"\r✅ Deskew job completed in {d_elapsed}s!{' ' * 20}\n")
                sys.stdout.flush()
                break
            elif d_fail.exists():
                sys.stdout.write(f"\r❌ Deskew job failed after {d_elapsed}s!{' ' * 20}\n")
                sys.stdout.flush()
                break
            
            # Live ticking timer for the GPUs
            sys.stdout.write(f"\r⏱️  Deskew Elapsed: {d_elapsed}s | Status: GPUs are crunching...{' ' * 5}")
            sys.stdout.flush()
            time.sleep(2)
            
    else:
        print("\n❌ Crop job failed. Aborting deskew submission.")
        print(f"   Check logs at: {fail_path}.log")


def _save_intermediate_crops(
    image_data: np.ndarray,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
    output_dir: Path,
    base_name: str,
    rotate_90: bool = False,
) -> None:
    """Save cropped ROI regions from napari memory into a CROP/ subfolder.

    This is a diagnostic tool for inspecting what the crop stage produces
    before deskewing. Saves T=0 only, one file per camera/ROI combination,
    using the same naming convention as the full pipeline output.
    """
    crop_dir = output_dir / "CROP"
    crop_dir.mkdir(parents=True, exist_ok=True)

    data = image_data
    ndim = data.ndim

    # Determine data layout and extract a single timepoint
    # Expected shapes: 5D (T,Z,C,Y,X), 4D (Z,C,Y,X), or 3D (Z,Y,X)
    if ndim == 5:
        data = data[0]  # T=0 → (Z,C,Y,X)
    if ndim >= 4 and data.ndim == 4:
        Z, C, Y, X = data.shape
    elif data.ndim == 3:
        # Single-channel: (Z,Y,X)
        Z, Y, X = data.shape
        C = 1
        data = data[:, np.newaxis, :, :]  # → (Z,1,Y,X)
    else:
        print(f"⚠️  Cannot save intermediates: unexpected data shape {image_data.shape}")
        return

    n_excitations = max(1, C // 2)
    tif_meta = {"axes": "ZYX"}
    saved_count = 0

    for exc in range(n_excitations):
        cam0_idx = exc
        cam1_idx = exc + n_excitations if C > 1 else 0
        out_base = exc * 4

        for label, roi, cam_idx, ch_offset in [
            ("Bot-Cam0", bottom_roi, cam0_idx, 0),
            ("Top-Cam0", top_roi, cam0_idx, 1),
            ("Top-Cam1", top_roi, cam1_idx, 2),
            ("Bot-Cam1", bottom_roi, cam1_idx, 3),
        ]:
            if cam_idx >= C:
                continue
            # Extract: data is (Z, C, Y, X)
            stack = data[:, cam_idx, roi[0], roi[1]]  # → (Z, crop_Y, crop_X)
            if rotate_90:
                stack = np.rot90(stack, k=1, axes=(1, 2))
            out_ch = out_base + ch_offset
            out_name = f"{base_name}_C{out_ch:02d}_T000.tif"
            tifffile.imwrite(
                crop_dir / out_name,
                stack,
                imagej=True,
                metadata=tif_meta,
            )
            saved_count += 1

    print(f"💾 Saved {saved_count} intermediate crops to {crop_dir}")


@magic_factory(
    call_button="🚀 Crop -> (Decon) -> Deskew",
    xy_pixel_size={"step": 0.001},
    z_step_um={"step": 0.001},
)
def petakit_pipeline(
    viewer: "napari.viewer.Viewer",
    image_layer: "napari.layers.Image",
    shapes_layer: "napari.layers.Shapes",
    z_step_um: float = 0.1,
    xy_pixel_size: float = 0.136,
    sheet_angle_deg: float = 30.0,
    output_format: str = "tiff-series",
    rotate_90: bool = True,
    save_intermediates: bool = True,
    objective_scan: bool = False,
    z_stage_scan: bool = False,
    reverse: bool = True,
    run_decon: bool = True,
    psf_path: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/Master_PSF_Final_Strict.tif"),
    decon_iters: int = 10,
):
    """Parses visual ROIs, enforces matching sizes, and chains crop+deskew jobs."""
    source_path = image_layer.source.path
    if not source_path:
        source_path = image_layer.metadata.get("path")
        
    if not source_path:
        print("❌ Error: Could not determine file path. Did you drag and drop?")
        return
    base_file = Path(source_path)

    if len(shapes_layer.data) != 2:
        print(
            f"❌ Error: Please draw exactly 2 rectangles (Top and Bottom). "
            f"You have {len(shapes_layer.data)}."
        )
        return

    rois = []
    for i, shape_coords in enumerate(shapes_layer.data):
        y_coords = shape_coords[:, -2]
        x_coords = shape_coords[:, -1]

        ymin, ymax = int(np.min(y_coords)), int(np.max(y_coords))
        xmin, xmax = int(np.min(x_coords)), int(np.max(x_coords))

        rois.append(
            {
                "index": i,
                "ymin": ymin,
                "ymax": ymax,
                "xmin": xmin,
                "xmax": xmax,
                "y_center": (ymin + ymax) / 2,
            }
        )

    rois.sort(key=lambda r: r["y_center"])
    top = rois[0]
    bot = rois[1]

    max_y, max_x = image_layer.data.shape[-2:]

    def _clamp_to_bounds(roi: dict, max_y: int, max_x: int) -> dict:
        """Safely clamp ROI coordinates to image bounds, preventing zero-area crops."""
        roi["ymin"] = max(0, min(roi["ymin"], max_y - 1))
        roi["ymax"] = max(roi["ymin"] + 1, min(roi["ymax"], max_y))
        roi["xmin"] = max(0, min(roi["xmin"], max_x - 1))
        roi["xmax"] = max(roi["xmin"] + 1, min(roi["xmax"], max_x))
        return roi

    # 1. Clamp top ROI first (it defines the target crop size)
    top = _clamp_to_bounds(top, max_y, max_x)
    height = top["ymax"] - top["ymin"]
    width = top["xmax"] - top["xmin"]

    # 2. Enforce same size on bottom ROI
    bot["ymax"] = bot["ymin"] + height
    bot["xmax"] = bot["xmin"] + width
    
    # 3. Clamp bottom ROI to prevent out-of-bounds overflow
    bot = _clamp_to_bounds(bot, max_y, max_x)

    top_roi = (slice(top["ymin"], top["ymax"]), slice(top["xmin"], top["xmax"]))
    bottom_roi = (slice(bot["ymin"], bot["ymax"]), slice(bot["xmin"], bot["xmax"]))

    # Update the shapes layer to reflect the clamped bottom ROI
    new_shapes = list(shapes_layer.data)
    new_bot_coords = np.array(
        [
            [bot["ymin"], bot["xmin"]],
            [bot["ymax"], bot["xmin"]],
            [bot["ymax"], bot["xmax"]],
            [bot["ymin"], bot["xmax"]],
        ]
    )
    new_shapes[bot["index"]] = new_bot_coords
    shapes_layer.data = new_shapes

    # 🔍 Metadata check & warning
    detected_z = _get_z_step_um(base_file)
    if detected_z is not None:
        print(f"🔍 Detected z-step from metadata: {detected_z} µm")
        if abs(detected_z - z_step_um) > 1e-6:
            print(f"⚠️  Widget value ({z_step_um} µm) differs from metadata! Update field if needed.")
    else:
        print("⚠️  Could not read z-step from metadata. Using widget value.")

    print(f"📁 Target: {base_file.name}")
    print(
        f"   Top ROI:    Y[{top_roi[0].start}:{top_roi[0].stop}], "
        f"X[{top_roi[1].start}:{top_roi[1].stop}]"
    )
    print(
        f"   Bottom ROI: Y[{bottom_roi[0].start}:{bottom_roi[0].stop}], "
        f"X[{bottom_roi[1].start}:{bottom_roi[1].stop}]"
    )

    # Derive output directory name
    folder_name = base_file.name
    if folder_name.lower().endswith(".ome.tif"):
        folder_name = folder_name[:-8]
    elif folder_name.lower().endswith(".tif"):
        folder_name = folder_name[:-4]
        
    t0_only = image_layer.metadata.get("t0_only", False)
    if t0_only:
        folder_name = f"{folder_name}_test"
        
    output_dir = base_file.parent / folder_name

    # Save intermediate crops for diagnostic inspection (before HPC job)
    if save_intermediates:
        try:
            _save_intermediate_crops(
                image_data=image_layer.data,
                top_roi=top_roi,
                bottom_roi=bottom_roi,
                output_dir=output_dir,
                base_name=folder_name,
                rotate_90=rotate_90,
            )
        except Exception as e:
            print(f"⚠️  Could not save intermediates: {e}")

    t0_only = image_layer.metadata.get("t0_only", False)
    target_timepoints = [1] if t0_only else None

    try:
        crop_job_ticket = submit_remote_crop_job(
            base_file=base_file,
            top_roi=top_roi,
            bottom_roi=bottom_roi,
            channels=None,
            timepoints=target_timepoints,
            output_format=output_format,
            rotate=rotate_90,
            z_step_um=z_step_um,
            xy_pixel_size=xy_pixel_size,
            test_mode=t0_only,
        )

        worker = _run_job_chain(
            crop_job_path=crop_job_ticket,
            output_dir=output_dir,
            z_step_um=z_step_um,
            rotate_90=rotate_90,
            sheet_angle_deg=sheet_angle_deg,
            objective_scan=objective_scan,
            z_stage_scan=z_stage_scan,
            reverse=reverse,
            run_decon=run_decon,
            psf_path=psf_path,
            decon_iters=decon_iters,
            gpu_decon=True,
        )
        worker.start()

    except Exception as e:
        print(f"❌ Failed to queue job pipeline: {e}")
@magic_factory(call_button="Load OME-TIFF")
def load_lazy_ome_tiff(file_path: Path, first_timepoint_only: bool = False) -> napari.types.LayerDataTuple:
    """Open a massive OME-TIFF as a lazy Dask array via a file selector."""
    store = tifffile.imread(file_path, aszarr=True)
    z = zarr.open(store, mode="r")
    lazy_data = da.from_zarr(z)

    layer_name = file_path.name
    if first_timepoint_only and lazy_data.ndim >= 5:
        lazy_data = lazy_data[0]
        layer_name = f"{file_path.name} (T=0)"

    return (lazy_data, {"name": layer_name, "multiscale": False, "metadata": {"path": str(file_path), "t0_only": first_timepoint_only}}, "image")


if __name__ == "__main__":
    viewer = napari.Viewer()

    viewer.add_shapes(
        name="Crop ROIs", edge_color="red", face_color="transparent", edge_width=5
    )

    viewer.window.add_dock_widget(
        petakit_pipeline(), name="Opym PetaKit", area="right"
    )

    viewer.window.add_dock_widget(
        load_lazy_ome_tiff(), name="OME-TIFF Loader", area="right"
    )

    napari.run()
