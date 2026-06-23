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


def _trim_tiff_files(output_dir: Path):
    """Zeroes out the coverslip artifact in the cropped TIFs before Decon."""
    import scipy.ndimage
    import skimage.measure
    
    tif_files = list(output_dir.glob("*.tif"))
    if not tif_files: return
    
    # Use the first TIF to calculate the trim bound
    first_tif = tifffile.imread(tif_files[0])
    t0_data = first_tif[0] if first_tif.ndim == 4 else first_tif
    
    smoothed = scipy.ndimage.gaussian_filter(t0_data, sigma=2)
    thresh = np.mean(smoothed) + 2 * np.std(smoothed)
    
    z_total_areas = np.zeros(smoothed.shape[0])
    for z in range(smoothed.shape[0]):
        mask = smoothed[z] > thresh
        labels = skimage.measure.label(mask)
        props = skimage.measure.regionprops(labels)
        if props:
            z_total_areas[z] = sum([p.area for p in props if p.area > 50])
            
    area_thresh = np.max(z_total_areas) * 0.33
    is_active = z_total_areas > area_thresh
    
    blocks = []
    in_block = False
    block_end = 0
    for z in range(len(is_active)-1, -1, -1):
        if is_active[z] and not in_block:
            in_block = True
            block_end = z
        elif not is_active[z] and in_block:
            in_block = False
            block_start = z + 1
            blocks.append((block_start, block_end))
    if in_block:
        blocks.append((0, block_end))
        
    longest_block = max(blocks, key=lambda b: b[1] - b[0])
    coverslip_z = longest_block[1]
    start_trim = coverslip_z + 2
    
    print(f"\n✂️ Auto-detected Coverslip Gap at Z={coverslip_z}. Cropping off Z={start_trim} onwards.")
    
    crop_dir = output_dir / "CROP"
    crop_dir.mkdir(exist_ok=True)
    
    for tif_path in tif_files:
        data = tifffile.imread(tif_path)
        if data.ndim == 3:
            cropped_data = data[:start_trim, :, :]
        elif data.ndim == 4:
            cropped_data = data[:, :start_trim, :, :]
        else:
            cropped_data = data
            
        out_path = crop_dir / tif_path.name
        tifffile.imwrite(out_path, cropped_data)
        
    print(f"✅ Successfully cropped the coverslip reflection and saved copies to {crop_dir}")
    return start_trim


def _auto_detect_rois(image_layer, needs_top=True, needs_bot=True) -> list[np.ndarray]:
    """Auto-detects the biological sample ROI from the top and bottom halves.
    Persists max_h and max_w to master_roi.json so future images match exactly.
    """
    import scipy.ndimage
    import skimage.measure
    import json
    from pathlib import Path
    
    print(f"\n🔍 Auto-detecting ROIs from data (Needs Top: {needs_top}, Needs Bot: {needs_bot})...")
    
    master_roi_path = None
    if image_layer.source and image_layer.source.path:
        master_roi_path = Path(image_layer.source.path).parent / "master_roi.json"
        
    master_h, master_w = 200, 200
    if master_roi_path and master_roi_path.exists():
        try:
            with open(master_roi_path, 'r') as f:
                d = json.load(f)
                master_h = d.get("max_h", 200)
                master_w = d.get("max_w", 200)
            print(f"   📂 Loaded previous bounds from master_roi.json (H: {master_h}, W: {master_w})")
        except: pass
        
    # Get T=0 data. For lazy dask arrays, .compute() pulls just that timepoint
    t0_data = image_layer.data[0] if hasattr(image_layer.data, "compute") else image_layer.data[0]
    if hasattr(t0_data, "compute"):
        print("   (Fetching T=0 chunk from Zarr...)")
        t0_data = t0_data.compute()
        
    print("   Projecting over Z and Channels...")
    # Max project over Z (axis 0) and Channels (axis 1) if dimensions match TZCYX
    if t0_data.ndim >= 3:
        # Assuming (Z, C, Y, X) or (Z, Y, X)
        proj_axes = tuple(range(t0_data.ndim - 2))
        max_proj = np.max(t0_data, axis=proj_axes)
    else:
        max_proj = t0_data
        
    max_y, max_x = max_proj.shape
    half_y = max_y // 2
    
    def _find_half_roi(image_half, offset_y=0):
        smoothed = scipy.ndimage.gaussian_filter(image_half, sigma=5)
        thresh = np.mean(smoothed) + 2 * np.std(smoothed)
        mask = smoothed > thresh
        
        labels = skimage.measure.label(mask)
        props = skimage.measure.regionprops(labels)
        
        if not props:
            return None
            
        min_row, min_col, max_row, max_col = image_half.shape[0], image_half.shape[1], 0, 0
        found = False
        
        for p in props:
            if p.area > 500: # Ignore tiny hot pixel clusters
                r0, c0, r1, c1 = p.bbox
                min_row = min(min_row, r0)
                min_col = min(min_col, c0)
                max_row = max(max_row, r1)
                max_col = max(max_col, c1)
                found = True
                
        if not found:
            return None
            
        # Add 50 pixel safety padding
        pad = 50
        min_row = max(0, min_row - pad)
        min_col = max(0, min_col - pad)
        max_row = min(image_half.shape[0], max_row + pad)
        max_col = min(image_half.shape[1], max_col + pad)
        
        return {
            "ymin": min_row + offset_y,
            "ymax": max_row + offset_y,
            "xmin": min_col,
            "xmax": max_col
        }

    top_roi_dict = _find_half_roi(max_proj[:half_y, :], 0) if needs_top else None
    bot_roi_dict = _find_half_roi(max_proj[half_y:, :], half_y) if needs_bot else None
    
    # Unify bounding box sizes so Top and Bottom have the exact same dimensions.
    # We take the maximum width and maximum height across both detections.
    valid_rois = [r for r in [top_roi_dict, bot_roi_dict] if r is not None]
    shapes = []
    
    if valid_rois:
        detected_max_h = max(r["ymax"] - r["ymin"] for r in valid_rois)
        detected_max_w = max(r["xmax"] - r["xmin"] for r in valid_rois)
        
        # 🛡️ Guardrails: The cell ROIs should typically be around 600x900
        EXPECTED_H = 600
        EXPECTED_W = 900
        
        if detected_max_h < 250 or detected_max_h > 1000:
            print(f"⚠️  Detected ROI Height ({detected_max_h}) is outside normal biological bounds. Clamping to {EXPECTED_H}.")
            detected_max_h = EXPECTED_H
            
        if detected_max_w < 400 or detected_max_w > 1800:
            print(f"⚠️  Detected ROI Width ({detected_max_w}) is outside normal biological bounds. Clamping to {EXPECTED_W}.")
            detected_max_w = EXPECTED_W
        
        # Enforce minimum sizes and blend with master.json if it existed
        max_h = max(detected_max_h, master_h)
        max_w = max(detected_max_w, master_w)
        
        # Save back the new maximums to ensure future images match
        if master_roi_path:
            try:
                with open(master_roi_path, 'w') as f:
                    json.dump({"max_h": max_h, "max_w": max_w}, f)
            except Exception as e:
                print(f"⚠️ Could not save master_roi.json: {e}")
                
        for r_dict in [top_roi_dict, bot_roi_dict]:
            if r_dict is None:
                continue
                
            # Extend upwards instead of centering
            new_ymax = r_dict["ymax"]
            new_ymin = max(0, new_ymax - max_h)
            
            x_center = (r_dict["xmin"] + r_dict["xmax"]) // 2
            new_xmin = max(0, x_center - max_w // 2)
            new_xmax = min(max_x, x_center + max_w // 2)
            
            print(f"   Unified ROI: Y[{new_ymin}:{new_ymax}], X[{new_xmin}:{new_xmax}]")
            coords = np.array([
                [new_ymin, new_xmin],
                [new_ymin, new_xmax],
                [new_ymax, new_xmax],
                [new_ymax, new_xmin],
            ])
            shapes.append(coords)
            
    if not shapes:
        print("❌ Could not detect any valid structures above background noise!")
        
    return shapes

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
    auto_trim_coverslip: bool = True,
    psf_paths: dict | None = None,
    decon_iters: int = 10,
    use_omw: bool = False,
    gpu_decon: bool = True,
    active_channels: list[str] | None = None,
    processing_metadata: dict | None = None,
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
                
                if auto_trim_coverslip:
                    print("\n✂️ Auto-Trimming coverslip artifact from final Deskewed TIFs...")
                    dsr_dir = deskew_input / "DSR"
                    if dsr_dir.exists():
                        start_trim = _trim_tiff_files(dsr_dir)
                        if processing_metadata and start_trim is not None:
                            processing_metadata["final_z_crop_start"] = 0
                            processing_metadata["final_z_crop_end"] = start_trim
                    else:
                        print(f"⚠️ DSR directory not found at {dsr_dir}. Skipping trim.")
                        
                if processing_metadata:
                    import json
                    target_dir = (deskew_input / "DSR" / "CROP") if auto_trim_coverslip else (deskew_input / "DSR")
                    if not target_dir.exists():
                        target_dir.mkdir(parents=True, exist_ok=True)
                    with open(target_dir / "processing_metadata.json", "w") as f:
                        json.dump(processing_metadata, f, indent=4)
                    print(f"✅ Saved sidecar processing metadata to {target_dir / 'processing_metadata.json'}")
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
    manual_z_step={"step": 0.01, "label": "Manual Z-Step (µm)"},
)
def petakit_pipeline(
    viewer: "napari.viewer.Viewer",
    image_layer: "napari.layers.Image",
    shapes_layer: "napari.layers.Shapes",
    manual_z_step: float = 0.3,
    output_format: str = "tiff-series",
    auto_trim_coverslip: bool = True,
    run_decon: bool = True,
    exposure_mode: typing.Literal["Single Exposure (All Lasers)", "Sequential Series"] = "Single Exposure (All Lasers)",
    psf_gfp: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_calcein_violet: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_mscarlet: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_cf647: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    psf_master: Path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"),
    decon_iters: int = 25,
    use_omw: bool = False,
):
    """Parses visual ROIs, enforces matching sizes, and chains crop+deskew jobs."""
    rotate_90 = True
    reverse = True
    objective_scan = False
    z_stage_scan = False
    
    xy_pixel_size = 0.136
    sheet_angle_deg = 60.0
    
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

    # 🔍 Auto-Detect ROIs if none are drawn
    if len(shapes_layer.data) == 0:
        needs_top = pipeline_widget.save_gfp.value or pipeline_widget.save_mscarlet.value
        needs_bot = pipeline_widget.save_calcein_violet.value or pipeline_widget.save_cf647.value
        
        detected_shapes = _auto_detect_rois(image_layer, needs_top=needs_top, needs_bot=needs_bot)
        if detected_shapes:
            shapes_layer.data = detected_shapes
            print("✏️  Drew auto-detected ROIs onto the Napari shapes layer.")
        else:
            print("⚠️  No ROIs drawn and auto-detection failed. Exiting.")
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

    # 🔍 AcqSettings check
    try:
        z_step_um = pipeline_widget.detected_z_step
    except Exception:
        z_step_um = None
        
    if z_step_um is None:
        z_step_um = manual_z_step
        print(f"⚠️ AcqSettings.txt not found. Using manual z-step: {z_step_um} µm")
    else:
        print(f"🔍 Using auto-detected z-step from AcqSettings.txt: {z_step_um} µm")

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

        processing_metadata = {
            "source_file": str(base_file),
            "xy_pixel_size": xy_pixel_size,
            "z_step_um": z_step_um,
            "sheet_angle_deg": sheet_angle_deg,
            "exposure_mode": exposure_mode,
            "rois": rois,
            "active_channels": active_channels,
            "run_decon": run_decon,
            "decon_iters": decon_iters,
            "use_omw": use_omw,
            "psf_paths": {k: str(v) for k, v in psf_paths_dict.items()} if psf_paths_dict else None,
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
            auto_trim_coverslip=auto_trim_coverslip,
            psf_paths=psf_paths_dict,
            decon_iters=decon_iters,
            use_omw=use_omw,
            gpu_decon=True,
            active_channels=active_channels,
            processing_metadata=processing_metadata,
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
        
    z_step = "Missing"
    acq_settings_path = file_path.parent / "AcqSettings.txt"
    detected_z_val = None
    if acq_settings_path.exists():
        import json
        try:
            with open(acq_settings_path, 'r') as f:
                settings = json.load(f)
                detected_z_val = float(settings.get("stepSizeUm", 0.3))
                z_step = str(detected_z_val)
                
                # Parse excitations
                channels = settings.get("channels", [])
                configs = [ch.get("config_", "") for ch in channels if ch.get("useChannel_", True)]
                
                has_488 = any("488" in c.lower() for c in configs)
                has_405 = any("405" in c.lower() for c in configs)
                has_561 = any("561" in c.lower() for c in configs)
                has_642 = any("642" in c.lower() or "640" in c.lower() for c in configs)
                has_all = any("all lasers" in c.lower() for c in configs)
                
                if has_all:
                    pipeline_widget.save_gfp.value = True
                    pipeline_widget.save_calcein_violet.value = True
                    pipeline_widget.save_mscarlet.value = True
                    pipeline_widget.save_cf647.value = True
                else:
                    pipeline_widget.save_gfp.value = has_488
                    pipeline_widget.save_calcein_violet.value = has_405
                    pipeline_widget.save_mscarlet.value = has_561
                    pipeline_widget.save_cf647.value = has_642
                    
                print(f"🔦 Parsed Excitation Configs: {configs}")
        except Exception as e: 
            print(f"⚠️ Error parsing AcqSettings.txt: {e}")
            
    try:
        if detected_z_val is not None:
            pipeline_widget.param_label.value = f"XY: 0.136 µm | Angle: 60.0° | Auto Z-Step: {z_step} µm"
            pipeline_widget.manual_z_step.visible = False
            pipeline_widget.detected_z_step = detected_z_val
        else:
            pipeline_widget.param_label.value = f"XY: 0.136 µm | Angle: 60.0° | Z-Step: ⚠️ MISSING"
            pipeline_widget.manual_z_step.visible = True
            pipeline_widget.detected_z_step = None
    except NameError:
        pass

    return (lazy_data, {"name": layer_name, "multiscale": False, "metadata": {"path": str(file_path), "t0_only": first_timepoint_only}}, "image")


@magic_factory(call_button="Trim Z-Range (Zero-out)", buffer_slices={"min": 0, "max": 100}, detection_threshold={"min": 0.01, "max": 1.0, "step": 0.05})
def trim_coverslip_reflection(
    layer: "napari.layers.Image",
    auto_detect: bool = True,
    z_start: int = 309,
    buffer_slices: int = 2,
    detection_threshold: float = 0.35,
) -> napari.types.LayerDataTuple:
    """Zeroes out the Z-slices below the coverslip to hide reflections."""
    import numpy as np
    data = layer.data
    
    if isinstance(data, np.ndarray):
        new_data = data.copy()
    else:
        new_data = np.array(data)
        
    if auto_detect:
        import scipy.ndimage
        import skimage.measure
        
        # Determine the coverslip slice from the first timepoint only
        t0_data = new_data[0] if new_data.ndim == 4 else new_data
        
        # 1. Apply a bit of smoothing to the data
        print("Auto-detect: Smoothing data...")
        smoothed = scipy.ndimage.gaussian_filter(t0_data, sigma=2)
        
        # 2. Use a smart threshold to isolate objects
        thresh = np.mean(smoothed) + 2 * np.std(smoothed)
        
        # 3. Find the total connected component area on each 2D plane
        print("Auto-detect: Finding connected components...")
        z_total_areas = np.zeros(smoothed.shape[0])
        for z in range(smoothed.shape[0]):
            mask = smoothed[z] > thresh
            labels = skimage.measure.label(mask)
            props = skimage.measure.regionprops(labels)
            if props:
                z_total_areas[z] = sum([p.area for p in props if p.area > 50])
                
        peak_z = int(np.argmax(z_total_areas))
        
        # A slice is considered "active" (has biological objects) if total area is > 33% of the peak area
        area_thresh = np.max(z_total_areas) * 0.33
        is_active = z_total_areas > area_thresh
        
        # Now we scan backwards from the absolute end (Z_max) down to 0, exactly as requested,
        # mapping out all the distinct structural blocks (e.g. reflection, cell, debris).
        blocks = []
        in_block = False
        block_end = 0
        for z in range(len(is_active)-1, -1, -1):
            if is_active[z] and not in_block:
                in_block = True
                block_end = z
            elif not is_active[z] and in_block:
                in_block = False
                block_start = z + 1
                blocks.append((block_start, block_end))
        if in_block:
            blocks.append((0, block_end))
            
        # The true cell is the thickest/longest contiguous block of biological objects!
        longest_block = max(blocks, key=lambda b: b[1] - b[0])
        coverslip_z = longest_block[1]
        
        # We trim exactly starting from the coverslip plus the user's buffer
        start_trim = coverslip_z + buffer_slices
        print(f"Auto-detected Coverslip Gap at Z={coverslip_z}. Trimming from Z={start_trim} onwards.")
    else:
        start_trim = z_start
        
    z_end = new_data.shape[-3]
        
    if new_data.ndim == 3:
        new_data[start_trim:z_end, :, :] = 0
    elif new_data.ndim == 4:
        new_data[:, start_trim:z_end, :, :] = 0
        
    return (new_data, {"name": f"{layer.name} (Trimmed)"}, "image")


if __name__ == "__main__":
    viewer = napari.Viewer()

    viewer.add_shapes(
        name="Crop ROIs", edge_color="red", face_color="transparent", edge_width=5
    )

    # Instantiate the pipeline widget globally so the function can access its custom layout values
    global pipeline_widget
    pipeline_widget = petakit_pipeline()
    pipeline_widget.detected_z_step = None
    pipeline_widget.manual_z_step.visible = False
    
    from magicgui.widgets import Container, CheckBox, Label
    param_label = Label(value="XY: 0.136 µm | Angle: 60.0° | Z-Step: (Load Data to Auto-Detect)")
    pipeline_widget.param_label = param_label
    pipeline_widget.insert(0, param_label)
    
    # Create the checkboxes manually since we removed them from the function signature
    from magicgui.widgets import Container, CheckBox, Label
    save_gfp = CheckBox(value=True, name="save_gfp", text="save gfp")
    save_calcein_violet = CheckBox(value=False, name="save_calcein_violet", text="save calcein violet")
    save_mscarlet = CheckBox(value=False, name="save_mscarlet", text="save mscarlet")
    save_cf647 = CheckBox(value=False, name="save_cf647", text="save cf647")
    
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
    
    # Hide all PSFs except GFP initially
    pipeline_widget.psf_calcein_violet.visible = False
    pipeline_widget.psf_mscarlet.visible = False
    pipeline_widget.psf_cf647.visible = False
    
    # Connect visibility toggles
    @save_gfp.changed.connect
    def _toggle_gfp(checked: bool):
        pipeline_widget.psf_gfp.visible = checked
        
    @save_calcein_violet.changed.connect
    def _toggle_calcein(checked: bool):
        pipeline_widget.psf_calcein_violet.visible = checked
        
    @save_mscarlet.changed.connect
    def _toggle_mscarlet(checked: bool):
        pipeline_widget.psf_mscarlet.visible = checked
        
    @save_cf647.changed.connect
    def _toggle_cf647(checked: bool):
        pipeline_widget.psf_cf647.visible = checked
    
    @pipeline_widget.use_omw.changed.connect
    def _on_omw_changed(checked: bool):
        if checked:
            pipeline_widget.decon_iters.value = 2
        else:
            pipeline_widget.decon_iters.value = 25

    viewer.window.add_dock_widget(
        load_lazy_ome_tiff(), name="OME-TIFF Loader", area="right"
    )

    viewer.window.add_dock_widget(
        pipeline_widget, name="Opym PetaKit", area="right"
    )

    viewer.window.add_dock_widget(
        trim_coverslip_reflection(), name="Trim Coverslip Artifact", area="right"
    )

    napari.run()
