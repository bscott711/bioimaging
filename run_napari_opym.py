import sys
import time
import typing
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
    crop_job_path: Path,
    output_dir: Path,
    z_step_um: float = 0.1,
    rotate_90: bool = True,
    sheet_angle_deg: float = 60.0,
    objective_scan: bool = False,
    z_stage_scan: bool = False,
    reverse: bool = True,
    run_decon: bool = True,
    psf_paths: dict | None = None,
    decon_iters: int = 10,
    use_omw: bool = False,
    gpu_decon: bool = True,
    active_channels: list[str] | None = None,
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

        if run_decon:
            print("\n🚀 Submitting parallel Decon jobs to PetaKit...")
            emissions = {
                "GFP": 525,
                "Calcein_Violet": 450,
                "mScarlet": 595,
                "CF647": 670,
            }
            
            if active_channels:
                emissions = {k: v for k, v in emissions.items() if k in active_channels}
            
            decon_tickets = []
            
            for name, wvl in emissions.items():
                if list(output_dir.glob(f"*{name}*.tif")):
                    psf_file = psf_paths.get(name) if psf_paths else Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif")
                    channel_pattern = f"{output_dir.name}_{name}"
                    ticket = submit_remote_decon_job(
                        input_target=output_dir,
                        psf_paths=psf_file,
                        iterations=decon_iters,
                        gpu_job=gpu_decon,
                        skewed=True,
                        result_dir_name="Decon",
                        channel_patterns=[channel_pattern],
                        rl_method="omw" if use_omw else "simplified",
                    )
                    decon_tickets.append(ticket)
            
            if not decon_tickets:
                master_psf = psf_paths.get("Master") if psf_paths else None
                if master_psf:
                    print("⚠️ No output crops found matching known emission wavelengths. Falling back to single master job...")
                    ticket = submit_remote_decon_job(
                        input_target=output_dir,
                        psf_paths=master_psf,
                        iterations=decon_iters,
                        gpu_job=gpu_decon,
                        skewed=True,
                        result_dir_name="Decon",
                        channel_patterns=[output_dir.name],
                        rl_method="omw" if use_omw else "simplified",
                    )
                    decon_tickets.append(ticket)
            
            print(f"🚀 Dispatched {len(decon_tickets)} Decon jobs concurrently!")
            
            d_start = time.time()
            decon_success = False
            
            while True:
                d_elapsed = int(time.time() - d_start)
                completed_count = 0
                failed_count = 0
                
                for t in decon_tickets:
                    if (base_dir / "completed" / t.name).exists():
                        completed_count += 1
                    elif (base_dir / "failed" / t.name).exists():
                        failed_count += 1
                
                if failed_count > 0:
                    sys.stdout.write(f"\r❌ Decon jobs failed after {d_elapsed}s!{' ' * 20}\n")
                    sys.stdout.flush()
                    break
                elif completed_count == len(decon_tickets) and len(decon_tickets) > 0:
                    sys.stdout.write(f"\r✅ All Decon jobs completed in {d_elapsed}s!{' ' * 20}\n")
                    sys.stdout.flush()
                    decon_success = True
                    break
                
                sys.stdout.write(f"\r⏱️  Decon Elapsed: {d_elapsed}s | Status: {completed_count}/{len(decon_tickets)} completed...{' ' * 5}")
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
    sheet_angle_deg: float = 60.0,
    output_format: str = "tiff-series",
    run_decon: bool = True,
    exposure_mode: typing.Literal["Single Exposure (All Lasers)", "Sequential Series"] = "Single Exposure (All Lasers)",
    psf_gfp: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_calcein_violet: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_mscarlet: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_cf647: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_master: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    decon_iters: int = 10,
    use_omw: bool = False,
):
    """Parses visual ROIs, enforces matching sizes, and chains crop+deskew jobs."""
    rotate_90 = True
    reverse = True
    objective_scan = False
    z_stage_scan = False
    
    source_path = image_layer.source.path
    if not source_path:
        source_path = image_layer.metadata.get("path")
        
    if not source_path:
        print("❌ Error: Could not determine file path. Did you drag and drop?")
        return
    base_file = Path(source_path)

    if len(shapes_layer.data) not in [1, 2]:
        print(
            f"❌ Error: Please draw exactly 1 or 2 rectangles. "
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

    max_y, max_x = image_layer.data.shape[-2:]

    top = None
    bot = None
    if len(rois) == 1:
        if rois[0]["y_center"] < max_y / 2:
            top = rois[0]
        else:
            bot = rois[0]
    else:
        rois.sort(key=lambda r: r["y_center"])
        top = rois[0]
        bot = rois[1]

    def _clamp_to_bounds(roi: dict, max_y: int, max_x: int) -> dict:
        """Safely clamp ROI coordinates to image bounds, preventing zero-area crops."""
        roi["ymin"] = max(0, min(roi["ymin"], max_y - 1))
        roi["ymax"] = max(roi["ymin"] + 1, min(roi["ymax"], max_y))
        roi["xmin"] = max(0, min(roi["xmin"], max_x - 1))
        roi["xmax"] = max(roi["xmin"] + 1, min(roi["xmax"], max_x))
        return roi

    top_roi = None
    bottom_roi = None
    height = 0
    width = 0

    if top:
        top = _clamp_to_bounds(top, max_y, max_x)
        height = top["ymax"] - top["ymin"]
        width = top["xmax"] - top["xmin"]
        top_roi = (slice(top["ymin"], top["ymax"]), slice(top["xmin"], top["xmax"]))

    new_shapes = list(shapes_layer.data)
    
    if bot:
        # Enforce same size as top ROI if both exist, otherwise take bot's drawn size
        if top:
            bot["ymax"] = bot["ymin"] + height
            bot["xmax"] = bot["xmin"] + width
            
        bot = _clamp_to_bounds(bot, max_y, max_x)
        bottom_roi = (slice(bot["ymin"], bot["ymax"]), slice(bot["xmin"], bot["xmax"]))

        new_bot_coords = np.array([
            [bot["ymin"], bot["xmin"]],
            [bot["ymin"], bot["xmax"]],
            [bot["ymax"], bot["xmax"]],
            [bot["ymax"], bot["xmin"]],
        ])
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
    if top_roi:
        print(f"   Top ROI:    Y[{top_roi[0].start}:{top_roi[0].stop}], X[{top_roi[1].start}:{top_roi[1].stop}]")
    if bottom_roi:
        print(f"   Bottom ROI: Y[{bottom_roi[0].start}:{bottom_roi[0].stop}], X[{bottom_roi[1].start}:{bottom_roi[1].stop}]")

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



    t0_only = image_layer.metadata.get("t0_only", False)
    target_timepoints = [1] if t0_only else None

    active_channels = []
    try:
        # If the widget exists globally, read from it directly
        if pipeline_widget.save_gfp.value: active_channels.append("GFP")
        if pipeline_widget.save_calcein_violet.value: active_channels.append("Calcein_Violet")
        if pipeline_widget.save_mscarlet.value: active_channels.append("mScarlet")
        if pipeline_widget.save_cf647.value: active_channels.append("CF647")
    except NameError:
        # Fallback if run outside of __main__
        active_channels = ["GFP", "Calcein_Violet", "mScarlet", "CF647"]

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
            exposure_mode=exposure_mode,
            active_channels=active_channels,
        )

        psf_paths_dict = {
            "GFP": psf_gfp,
            "Calcein_Violet": psf_calcein_violet,
            "mScarlet": psf_mscarlet,
            "CF647": psf_cf647,
            "Master": psf_master,
        }

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
            psf_paths=psf_paths_dict,
            decon_iters=decon_iters,
            use_omw=use_omw,
            gpu_decon=True,
            active_channels=active_channels,
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

    # Instantiate the pipeline widget globally so the function can access its custom layout values
    global pipeline_widget
    pipeline_widget = petakit_pipeline()
    
    # Create the checkboxes manually since we removed them from the function signature
    from magicgui.widgets import Container, CheckBox, Label
    save_gfp = CheckBox(value=True, name="save_gfp", text="save gfp")
    save_calcein_violet = CheckBox(value=True, name="save_calcein_violet", text="save calcein violet")
    save_mscarlet = CheckBox(value=True, name="save_mscarlet", text="save mscarlet")
    save_cf647 = CheckBox(value=True, name="save_cf647", text="save cf647")
    
    # Attach them to the widget object so the function can find them
    pipeline_widget.save_gfp = save_gfp
    pipeline_widget.save_calcein_violet = save_calcein_violet
    pipeline_widget.save_mscarlet = save_mscarlet
    pipeline_widget.save_cf647 = save_cf647
    
    # Restructure save boxes into a 2x2 grid
    space_1 = Label(value="    ")
    space_2 = Label(value="    ")
    
    save_row_1 = Container(layout="horizontal", widgets=[save_gfp, space_1, save_calcein_violet], labels=False)
    save_row_2 = Container(layout="horizontal", widgets=[save_mscarlet, space_2, save_cf647], labels=False)
    save_grid = Container(layout="vertical", widgets=[save_row_1, save_row_2], labels=False)
    
    # Insert the grid before the PSF path widgets
    idx = pipeline_widget.index(pipeline_widget.psf_gfp)
    pipeline_widget.insert(idx, save_grid)

    viewer.window.add_dock_widget(
        load_lazy_ome_tiff(), name="OME-TIFF Loader", area="right"
    )

    viewer.window.add_dock_widget(
        pipeline_widget, name="Opym PetaKit", area="right"
    )

    napari.run()
