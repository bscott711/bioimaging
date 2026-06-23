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
    submit_pipeline_job,
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
            
        # We don't add massive padding here, we just return the tight bounding box of the signal.
        # The unifying step will expand this to the full optical FOV.
        return {
            "ymin": min_row + offset_y,
            "ymax": max_row + offset_y,
            "xmin": min_col,
            "xmax": max_col
        }

    top_roi_dict = _find_half_roi(max_proj[:half_y, :], 0) if needs_top else None
    bot_roi_dict = _find_half_roi(max_proj[half_y:, :], half_y) if needs_bot else None
    
    # Unify bounding box sizes so Top and Bottom have the exact same dimensions.
    # We take the center of the detected signal, and expand it to capture the full optical FOV.
    valid_rois = [r for r in [top_roi_dict, bot_roi_dict] if r is not None]
    shapes = []
    
    if valid_rois:
        detected_max_h = max(r["ymax"] - r["ymin"] for r in valid_rois)
        detected_max_w = max(r["xmax"] - r["xmin"] for r in valid_rois)
        
        # The full illuminated optical FOV. 
        # Using 576x1152 (18x32 blocks of 64) for mathematically optimal CUDA/FFT performance!
        EXPECTED_H = 576
        EXPECTED_W = 1152
        
        # Enforce that our box is AT LEAST the expected optical FOV size
        max_h = max(detected_max_h, EXPECTED_H)
        max_w = max(detected_max_w, EXPECTED_W)
        
        if master_roi_path:
            try:
                with open(master_roi_path, 'w') as f:
                    json.dump({"max_h": max_h, "max_w": max_w}, f)
            except Exception as e:
                print(f"⚠️ Could not save master_roi.json: {e}")
                
        for r_dict in [top_roi_dict, bot_roi_dict]:
            if r_dict is None:
                continue
                
            # Find the center of the detected signal
            y_center = (r_dict["ymin"] + r_dict["ymax"]) // 2
            x_center = (r_dict["xmin"] + r_dict["xmax"]) // 2
            
            # Expand outwards from the center to capture the full optical FOV
            new_ymin = max(0, y_center - max_h // 2)
            new_ymax = new_ymin + max_h
            
            new_xmin = max(0, x_center - max_w // 2)
            new_xmax = min(max_x, new_xmin + max_w)
            
            # If expanding pushed us past the right edge, shift back left
            if new_xmax > max_x:
                new_xmax = max_x
                new_xmin = max(0, new_xmax - max_w)
                
            print(f"   Optical FOV ROI: Y[{new_ymin}:{new_ymax}], X[{new_xmin}:{new_xmax}]")
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

_global_z = None
_global_store = None

def _worker_init(base_file_str):
    global _global_z, _global_store
    import tifffile, zarr
    _global_store = tifffile.imread(base_file_str, aszarr=True)
    _global_z = zarr.open(_global_store, mode='r')

def _process_chunk_worker(t, c, is_top, roi, base_file, output_dir, psf_file, z_step_um, sheet_angle_deg, decon_iters, z_crop_end=None, c_axis=1, lazy_array=None):
    import tifffile
    import zarr
    import time
    import numpy as np
    from pathlib import Path
    from opym.petakit import submit_pipeline_job
    
    prefix = base_file.name
    if prefix.lower().endswith(".ome.tif"):
        prefix = prefix[:-8]
    elif prefix.lower().endswith(".tif"):
        prefix = prefix[:-4]
        
    shm_path = Path(f"/dev/shm/opym_jobs/{prefix}_T{t:04d}_C{c:02d}.ome.tif")
    out_sub_dir = output_dir
    
    shm_dir = Path("/dev/shm/opym_jobs")
    while shm_dir.exists() and len(list(shm_dir.glob("*.tif"))) > 25:
        time.sleep(1)
        
    global _global_z
    if _global_z is not None:
        z = _global_z
    elif lazy_array is not None:
        z = lazy_array
    else:
        store = tifffile.imread(str(base_file), aszarr=True)
        z = zarr.open(store, mode='r')
    
    if z.ndim == 5:
        if c_axis == 1:
            cropped = z[t, c, :, roi[0], roi[1]]
        else:
            cropped = z[t, :, c, roi[0], roi[1]]
    elif z.ndim == 4:
        if c_axis == 1:
            cropped = z[c, :, roi[0], roi[1]]
        else:
            cropped = z[:, c, roi[0], roi[1]]
    else:
        raise RuntimeError(f"❌ Extraction failed! Expected lazy_array to be 4D or 5D, but got {z.ndim}D array with shape {z.shape}.")

    shm_path.parent.mkdir(exist_ok=True, parents=True)
    tifffile.imwrite(shm_path, np.array(cropped), imagej=True)
    
    out_sub_dir.mkdir(exist_ok=True, parents=True)
    output_file = out_sub_dir / shm_path.name
    
    ticket = submit_pipeline_job(
        output_file=output_file,
        shm_path=shm_path,
        psf_paths=[str(psf_file)] if psf_file else [],
        z_step_um=z_step_um,
        iterations=decon_iters,
        sheet_angle_deg=sheet_angle_deg,
        interp_method="linear",
        z_crop_end=z_crop_end,
    )
    return ticket

@thread_worker(progress={'total': 0, 'desc': 'Extracting Zarr...'})
def _run_unified_job_chain(
    base_file: Path,
    output_dir: Path,
    top_roi: tuple | None,
    bottom_roi: tuple | None,
    z_step_um: float = 0.1,
    sheet_angle_deg: float = 60.0,
    psf_paths: dict | None = None,
    decon_iters: int = 10,
    active_channels: list[str] | None = None,
    processing_metadata: dict | None = None,
    target_timepoints: list[int] | None = None,
    master_roi_path: Path | None = None,
    lazy_array=None,
):
    import tifffile
    import zarr
    import time
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from run_pipeline_cli import get_extraction_plan
    
    print(f"\n🚀 Initializing Unified GPU Pipeline for {base_file}")
    
    store = tifffile.imread(str(base_file), aszarr=True)
    z = zarr.open(store, mode='r')
    
    num_t = z.shape[0]
    timepoints_to_run = target_timepoints if target_timepoints is not None else range(num_t)
    
    if len(z.shape) == 5:
        if z.shape[1] < z.shape[2]:
            num_c = z.shape[1]
            c_axis = 1
        else:
            num_c = z.shape[2]
            c_axis = 2
    else:
        num_c = 1
        c_axis = 1
        
    print(f"Detected Zarr shape: {z.shape}. C={num_c}. Extracting timepoints: {list(timepoints_to_run)}...")
    del z
    del store
    
    plan = get_extraction_plan(base_file)
    if not plan:
        plan = []
        for c in range(num_c):
            plan.append((c, True, f"C{c}"))
            plan.append((c, False, f"C{c}"))
            
    final_plan = []
    if active_channels is None: active_channels = []
    for c, is_top, laser_name in plan:
        if laser_name == "488nm" and "GFP" not in active_channels: continue
        if laser_name == "405nm" and "Calcein_Violet" not in active_channels: continue
        if laser_name == "561nm" and "mScarlet" not in active_channels: continue
        if laser_name == "640nm" and "CF647" not in active_channels: continue
        final_plan.append((c, is_top, laser_name))
        
    psf_map = {}
    if psf_paths:
        psf_map = {
            "488nm": psf_paths.get("488nm") or psf_paths.get("GFP"),
            "405nm": psf_paths.get("405nm") or psf_paths.get("Calcein_Violet"),
            "561nm": psf_paths.get("561nm") or psf_paths.get("mScarlet"),
            "640nm": psf_paths.get("640nm") or psf_paths.get("CF647"),
            "Master": psf_paths.get("Master")
        }
        
    # Fallback to old names if fallback plan
    if not plan and psf_paths:
        psf_map = {"C0": psf_map.get("488nm")} # Generic mapping
        
    def _wait_for_tickets(tickets_list, phase_name="chunks", custom_start_time=None):
        import sys
        base_dir = Path("~/petakit_jobs").expanduser()
        d_start = custom_start_time if custom_start_time is not None else time.time()
        while True:
            d_elapsed = int(time.time() - d_start)
            completed_count = 0
            failed_count = 0
            for tk in tickets_list:
                if (base_dir / "completed" / tk.name).exists() or (base_dir / "completed" / f".active_{tk.name}").exists():
                    completed_count += 1
                elif (base_dir / "failed" / tk.name).exists() or (base_dir / "failed" / f".active_{tk.name}").exists():
                    failed_count += 1
            if failed_count > 0:
                sys.stdout.write(f"\r❌ {phase_name} failed after {d_elapsed}s!{' ' * 20}\n")
                sys.stdout.flush()
                return False
            elif completed_count == len(tickets_list) and len(tickets_list) > 0:
                sys.stdout.write(f"\r✅ All {len(tickets_list)} {phase_name} completed in {d_elapsed}s!{' ' * 20}\n")
                sys.stdout.flush()
                return True
            sys.stdout.write(f"\r⏱️  {phase_name.capitalize()} Elapsed: {d_elapsed}s | Status: {completed_count}/{len(tickets_list)} completed...{' ' * 5}")
            sys.stdout.flush()
            time.sleep(2)
            
    start_trim = None
    master_dict = {}
    if master_roi_path and master_roi_path.exists():
        try:
            with open(master_roi_path, 'r') as f:
                master_dict = json.load(f)
                if "z_crop_end" in master_dict:
                    start_trim = master_dict["z_crop_end"]
                    print(f"   📂 Loaded previous unified Z-Trim from master_roi.json (Z=0 to Z={start_trim})")
        except Exception as e:
            print(f"⚠️ Could not parse master_roi.json: {e}")
            
    t_zero = timepoints_to_run[0]
    
    calibrator_tuples = []
    if start_trim is None:
        calibrator_tuples = [p for p in final_plan if p[2] == "488nm"]
        if not calibrator_tuples and final_plan:
            calibrator_tuples = [final_plan[0]]
            
        print(f"   🔍 Selected {calibrator_tuples} as primary calibrator for Z-trim.")
                    
        t0_tickets = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for c, is_top, laser_name in calibrator_tuples:
                roi = top_roi if is_top else bottom_roi
                if not roi: continue
                psf_file = psf_map.get(laser_name)
                futures.append(executor.submit(_process_chunk_worker, t_zero, c, is_top, roi, base_file, output_dir, psf_file, z_step_um, sheet_angle_deg, decon_iters, None, c_axis, lazy_array))
            for f in futures:
                res = f.result()
                if res: t0_tickets.append(res)
                
        print(f"\n🚀 Submitted T={t_zero} calibrator to determine unified Z-trim...")
        if not _wait_for_tickets(t0_tickets, phase_name=f"T={t_zero} calibrator chunks"):
            return
            
        print("\n✂️ Auto-Trimming coverslip artifact on T=0 calibrator...")
        start_trim = _trim_tiff_files(output_dir)
                
        if start_trim is not None:
            print(f"✅ Unified Z-Trim established: Z=0 to Z={start_trim}. Applying to all remaining timepoints!")
            if master_roi_path:
                master_dict["z_crop_end"] = start_trim
                try:
                    with open(master_roi_path, 'w') as f:
                        json.dump(master_dict, f)
                except Exception as e:
                    pass
        else:
            print("⚠️ Could not detect coverslip on T=0. Proceeding without unified Z-trim.")
            
    if processing_metadata and start_trim is not None:
        processing_metadata["final_z_crop_start"] = 0
        processing_metadata["final_z_crop_end"] = start_trim
        
    tickets = []
    global_start = time.time()
    
    from concurrent.futures import ProcessPoolExecutor
    
    with ProcessPoolExecutor(max_workers=4, initializer=_worker_init, initargs=(str(base_file),)) as executor:
        futures = []
        for t in timepoints_to_run:
            for c, is_top, laser_name in final_plan:
                if t == t_zero and (c, is_top, laser_name) in calibrator_tuples:
                    continue  # Already processed during calibrator phase
                
                roi = top_roi if is_top else bottom_roi
                if not roi: continue
                psf_file = psf_map.get(laser_name)
                # Pass lazy_array=None because zarr objects cannot be pickled across processes
                futures.append(executor.submit(_process_chunk_worker, t, c, is_top, roi, base_file, output_dir, psf_file, z_step_um, sheet_angle_deg, decon_iters, start_trim, c_axis, None))
        
        for f in futures:
            res = f.result()
            if res:
                tickets.append(res)
                
        print(f"\n🚀 {len(tickets)} remaining chunks extracted and submitted with Z-trim applied!")
        if tickets:
            _wait_for_tickets(tickets, phase_name="remaining chunks", custom_start_time=global_start)
        
        print("\n✅ All requested timepoints have been processed!")

    if processing_metadata:
        import json
        meta_dir = output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "processing_metadata.json", "w") as f:
            json.dump(processing_metadata, f, indent=4)



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
    
    if image_layer is None:
        from napari.utils.notifications import show_warning
        show_warning("Please load or select an Image layer first!")
        print("❌ Error: No Image Layer selected.")
        return
        
    source_path = image_layer.source.path
    if not source_path:
        source_path = image_layer.metadata.get("path")
        
    if not source_path:
        print("❌ Error: Could not determine file path. Did you drag and drop?")
        return
    base_file = Path(source_path)

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
    folder_name = "Decon" if run_decon else "DSR"
    t0_only = image_layer.metadata.get("t0_only", False)
    if t0_only:
        folder_name += "_test"
        
    output_dir = base_file.parent / folder_name



    t0_only = image_layer.metadata.get("t0_only", False)
    target_timepoints = [0] if t0_only else None

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
        psf_paths_dict = {
            "488nm": psf_gfp,
            "405nm": psf_calcein_violet,
            "561nm": psf_mscarlet,
            "640nm": psf_cf647,
            "Master": psf_master
        }
        
        processing_metadata = {
            "source_file": str(base_file),
            "z_step_um": z_step_um,
            "decon_iters": decon_iters,
            "active_channels": active_channels,
        }
        
        master_roi_path = base_file.parent / "master_roi.json"
        
        worker = _run_unified_job_chain(
            base_file=base_file,
            output_dir=output_dir,
            top_roi=top_roi,
            bottom_roi=bottom_roi,
            z_step_um=z_step_um,
            sheet_angle_deg=sheet_angle_deg,
            psf_paths=psf_paths_dict,
            decon_iters=decon_iters,
            active_channels=active_channels,
            processing_metadata=processing_metadata,
            target_timepoints=target_timepoints,
            master_roi_path=master_roi_path,
            lazy_array=image_layer.data,
        )
        worker.start()
    except Exception as e:
        print(f"❌ Failed to queue job pipeline: {e}")
@magic_factory(call_button="Load OME-TIFF")
def load_lazy_ome_tiff(file_path: Path, first_timepoint_only: bool = False) -> napari.types.LayerDataTuple:
    """Open a massive OME-TIFF as a lazy Dask array via a file selector."""
    import warnings
    import tifffile
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
                # Auto-check boxes based on actual analysis pipeline
                from run_pipeline_cli import get_extraction_plan
                try:
                    plan = get_extraction_plan(file_path)
                    if plan:
                        lasers = [p[2] for p in plan]
                        pipeline_widget.save_gfp.value = "488nm" in lasers
                        pipeline_widget.save_calcein_violet.value = "405nm" in lasers
                        pipeline_widget.save_mscarlet.value = "561nm" in lasers
                        pipeline_widget.save_cf647.value = "640nm" in lasers
                        print(f"🔦 Automatically checked channels based on analyze_channels pipeline: {lasers}")
                except Exception as e:
                    print(f"⚠️ Error using analyze_channels pipeline: {e}")
                    
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

    ome_dock = viewer.window.add_dock_widget(
        load_lazy_ome_tiff(), name="OME-TIFF Loader", area="left"
    )
    
    # Move OME-TIFF Loader to the upper left, above the layer controls, and layer list at bottom
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget
        qt_win = viewer.window._qt_window
        layer_controls = [d for d in qt_win.findChildren(QDockWidget) if d.objectName() == 'layer controls']
        layer_list = [d for d in qt_win.findChildren(QDockWidget) if d.objectName() == 'layer list']
        if layer_controls and layer_list:
            # First split: ome_dock is top, layer_controls is bottom
            qt_win.splitDockWidget(ome_dock, layer_controls[0], Qt.Vertical)
            # Second split: layer_controls is top, layer_list is bottom
            qt_win.splitDockWidget(layer_controls[0], layer_list[0], Qt.Vertical)
    except Exception as e:
        print(f"⚠️ Could not adjust left dock widget order: {e}")

    viewer.window.add_dock_widget(
        pipeline_widget, name="Opym PetaKit", area="right"
    )

    # viewer.window.add_dock_widget(
    #     trim_coverslip_reflection(), name="Trim Coverslip Artifact", area="bottom"
    # )

    napari.run()
