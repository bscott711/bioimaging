#!/usr/bin/env python3
import argparse
import time
import json
import numpy as np
import tifffile
import zarr
import scipy.ndimage
import skimage.measure
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import math
import multiprocessing as mp

import sys
# Make sure we can import opym from local projects
sys.path.append(str(Path.home() / "projects" / "opym_local" / "src"))
from opym.petakit import submit_pipeline_job

def auto_detect_rois(z_array, master_roi_path=None):
    print("\n🔍 Auto-detecting ROIs from T=0...")
    
    master_h, master_w = 200, 200
    if master_roi_path and master_roi_path.exists():
        try:
            with open(master_roi_path, 'r') as f:
                d = json.load(f)
                master_h = d.get("max_h", 200)
                master_w = d.get("max_w", 200)
            print(f"   📂 Loaded previous bounds from master_roi.json (H: {master_h}, W: {master_w})")
        except: pass

    print("   Fetching T=0 chunk from Zarr...")
    if z_array.ndim == 5:
        # Usually (T, C, Z, Y, X) or (T, Z, C, Y, X). Let's take T=0, C/Z=0
        t0_data = z_array[0, 0]
    elif z_array.ndim == 4:
        t0_data = z_array[0]
    else:
        t0_data = z_array
        
    print("   Projecting over Z and Channels...")
    if t0_data.ndim >= 3:
        proj_axes = tuple(range(t0_data.ndim - 2))
        max_proj = np.max(t0_data, axis=proj_axes)
    else:
        max_proj = t0_data
        
    max_y, max_x = max_proj.shape
    half_y = max_y // 2
    
def auto_detect_rois(z, master_roi_path=None):
    import numpy as np
    from skimage.measure import regionprops, label
    import json
    import scipy.ndimage
    import skimage.measure
    
    # Fast projection of middle timepoint to find ROIs
    t_mid = z.shape[0] // 2
    if len(z.shape) == 5:
        # z shape might be (T,C,Z,Y,X) or (T,Z,C,Y,X)
        if z.shape[1] < z.shape[2]: # (T,C,Z,Y,X)
            img = z[t_mid, 0, z.shape[2]//2, :, :]
        else: # (T,Z,C,Y,X)
            img = z[t_mid, z.shape[1]//2, 0, :, :]
    else:
        return None, None
        
    max_proj = np.array(img)
    max_y, max_x = max_proj.shape
    half_y = max_y // 2
    
    def _find_half_roi(image_half, offset_y=0):
        # Apply massive blur to turn the image into a smooth heat map
        smoothed = scipy.ndimage.gaussian_filter(image_half, sigma=20)
        
        # Find the coordinates of the absolute maximum intensity
        y_peak, x_peak = np.unravel_index(np.argmax(smoothed, axis=None), smoothed.shape)
        
        return {
            "y_center": int(y_peak + offset_y),
            "x_center": int(x_peak)
        }

    top_roi_dict = _find_half_roi(max_proj[:half_y, :], 0)
    bot_roi_dict = _find_half_roi(max_proj[half_y:, :], half_y)
    
    valid_rois = [r for r in [top_roi_dict, bot_roi_dict] if r is not None]
    rois_out = []
    
    if valid_rois:
        EXPECTED_H = 576
        EXPECTED_W = 1152
            
        import scipy.fft
        max_h = scipy.fft.next_fast_len(EXPECTED_H)
        max_w = scipy.fft.next_fast_len(EXPECTED_W)
        
        if master_roi_path:
            try:
                with open(master_roi_path, 'w') as f:
                    json.dump({"max_h": max_h, "max_w": max_w}, f)
            except: pass
            
        for r_dict in [top_roi_dict, bot_roi_dict]:
            if r_dict is None:
                rois_out.append(None)
                continue
                
            y_center = r_dict["y_center"]
            x_center = r_dict["x_center"]
            
            new_ymin = max(0, y_center - max_h // 2)
            new_ymax = new_ymin + max_h
            
            new_xmin = max(0, x_center - max_w // 2)
            new_xmax = min(max_x, new_xmin + max_w)
            
            if new_xmax > max_x:
                new_xmax = max_x
                new_xmin = max(0, new_xmax - max_w)
            
            print(f"   Optical FOV ROI: Y[{new_ymin}:{new_ymax}], X[{new_xmin}:{new_xmax}]")
            rois_out.append((slice(new_ymin, new_ymax), slice(new_xmin, new_xmax)))
    
    if not rois_out:
        print("❌ Could not detect any valid structures!")
        
    # Return (top_roi, bot_roi)
    if len(rois_out) == 1:
        return rois_out[0], None
    elif len(rois_out) >= 2:
        return rois_out[0], rois_out[1]
    return None, None

def get_z_step(target_path: Path):
    acq_settings = target_path.parent / "AcqSettings.txt"
    if acq_settings.exists():
        with open(acq_settings, 'r') as f:
            d = json.load(f)
            return float(d.get("stepSizeUm", 0.3))
    return 0.3

def get_extraction_plan(target_path: Path):
    from psf_tools.analyze_channels import analyze_directory, parse_name_for_fluorophores
    base_dir = target_path.parent.parent
    results = analyze_directory(str(base_dir))
    folder_name = target_path.parent.name
    
    match = next((r for r in results if r["Folder"] == folder_name), None)
    if not match: return []
    
    detected = match["Detected_Channels"]
    if not detected:
        detected = parse_name_for_fluorophores(base_dir.name)
        
    raw_configs = match["Raw_Configs"]
    
    has_488 = any("488" in c for c in detected)
    has_405 = any("405" in c for c in detected)
    has_561 = any("561" in c for c in detected)
    has_640 = any("640" in c for c in detected)
    
    plan = []
    out_c = 0
    
    if len(raw_configs) <= 1:
        if has_488: 
            plan.append((0, True, "488nm", out_c))
            out_c += 1
        if has_405: 
            plan.append((0, False, "405nm", out_c))
            out_c += 1
        if has_561: 
            plan.append((1, True, "561nm", out_c))
            out_c += 1
        if has_640: 
            plan.append((1, False, "640nm", out_c))
            out_c += 1
    else:
        for i, config in enumerate(raw_configs):
            config_lower = config.lower()
            if "488" in config_lower: 
                plan.append((i*2 + 0, True, "488nm", out_c))
                out_c += 1
            if "405" in config_lower: 
                plan.append((i*2 + 0, False, "405nm", out_c))
                out_c += 1
            if "561" in config_lower: 
                plan.append((i*2 + 1, True, "561nm", out_c))
                out_c += 1
            if "640" in config_lower: 
                plan.append((i*2 + 1, False, "640nm", out_c))
                out_c += 1
            
    return plan

def process_chunk(t_start, t_end, target_path, c_axis, extraction_plan, top_roi, bot_roi, base_name, output_dir_base, psf_map, z_step_um, debug, worker_id):
    """
    Process a contiguous chunk of timepoints directly from GPFS.
    Opens the multi-file Zarr store ONCE (paying a one-time metadata tax),
    then streams sequentially through the assigned range.
    """
    import tifffile, zarr, time, numpy as np
    from pathlib import Path
    from opym.petakit import submit_pipeline_job
    
    # Note: tifffile memory-maps by default when possible. The boundary delays
    # are due to GPFS file open/close token overhead when crossing 4GB splits.
    store = tifffile.imread(str(target_path), aszarr=True)
    z = zarr.open(store, mode='r')
    
    print(f"[Worker {worker_id:02d}] Initialized. Processing T={t_start} to {t_end-1}")
    
    for t in range(t_start, t_end):
        for c, is_top, laser_name, output_c in extraction_plan:
            roi = top_roi if is_top else bot_roi
            if not roi: continue
            
            shm_path = Path(f"/dev/shm/opym_jobs/{base_name}_T{t:04d}_C{output_c}.zarr")
            st = time.time()
            
            # BULK SINGLE-READ
            if z.ndim == 5:
                if c_axis == 1: cropped = np.array(z[t, c, :, roi[0], roi[1]])
                else: cropped = np.array(z[t, :, c, roi[0], roi[1]])
            else: continue
            
            extract_time = time.time() - st
            
            shm_path.parent.mkdir(exist_ok=True, parents=True)
            zarr.save(shm_path, cropped)
            
            output_file = output_dir_base / shm_path.name
            psf_path = psf_map.get(laser_name, None)
            
            submit_pipeline_job(
                output_file=output_file, shm_path=shm_path,
                psf_paths=[psf_path] if psf_path else None,
                z_step_um=z_step_um, iterations=10, debug=debug
            )
            
            print(f"[Worker {worker_id:02d}] T={t:03d} C={output_c} | Extract: {extract_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="End-to-End GPU Pipeline CLI")
    parser.add_argument("data_path", type=str, help="Path to the directory containing the ome.tif file")
    parser.add_argument("--psf", type=str, default="/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif", help="Path to PSF")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and profiling")
    parser.add_argument("--t0-only", action="store_true", help="Process only the first timepoint (T=0) for rapid testing")
    args = parser.parse_args()

    data_dir = Path(args.data_path).resolve()
    
    # Find the ome.tif file
    ome_tifs = list(data_dir.glob("*.ome.tif"))
    if not ome_tifs:
        print(f"No .ome.tif found in {data_dir}")
        return
        
    # Always pick the root file (shortest name), e.g. cell_MMStack_Pos0.ome.tif
    target_path = min(ome_tifs, key=lambda p: len(p.name))
    master_roi_path = data_dir / "master_roi.json"
    z_step_um = get_z_step(target_path)
    
    print(f"🚀 Initializing Pipeline for {target_path}")
    print("Parsing IFDs and creating Zarr store (Zero I/O overhead)...")
    st = time.time()
    store = tifffile.imread(str(target_path), aszarr=True)
    z = zarr.open(store, mode='r')
    print(f"✅ Parsed in {time.time()-st:.2f}s. Zarr shape: {z.shape}")
    
    top_roi, bot_roi = auto_detect_rois(z, master_roi_path=master_roi_path)
    
    num_t = z.shape[0]
    
    # Determine which axis is C and which is Z
    if len(z.shape) == 5:
        if z.shape[1] < z.shape[2]:
            # (T, C, Z, Y, X)
            num_c = z.shape[1]
            c_axis = 1
        else:
            # (T, Z, C, Y, X)
            num_c = z.shape[2]
            c_axis = 2
    else:
        num_c = 1
        c_axis = 1
        
    extraction_plan = get_extraction_plan(target_path)
    if not extraction_plan:
        print("⚠️ Warning: Could not generate extraction plan from analyze_channels.py! Defaulting to all channels.")
        # Fallback if parser fails
        out_c = 0
        for c in range(num_c):
            extraction_plan.append((c, True, f"C{c}", out_c))
            out_c += 1
            extraction_plan.append((c, False, f"C{c}", out_c))
            out_c += 1

    if args.t0_only:
        print("⚠️ --t0-only flag detected: Restricting pipeline to T=0.")
        num_t = min(1, num_t)
            
    print(f"Detected Zarr shape: {z.shape}. T={num_t}, C={num_c}.")
    print(f"Extraction Plan: {extraction_plan}")
    
    output_dir_base = data_dir / "Decon"
    psf_map = {
        "488nm": args.psf,
        "405nm": args.psf,
        "561nm": args.psf,
        "640nm": args.psf
    }
    
    # Release the global store so workers don't inherit a locked file handle
    del z
    del store
    
    # Clean the filename up for the workers to use
    import re
    from concurrent.futures import ProcessPoolExecutor
    base_name = re.sub(r'_\d+\.ome$', '', target_path.stem)
    
    print(f"\n🚀 Starting Direct GPFS Streaming (8 Workers)...")
    
    num_workers = min(8, mp.cpu_count() // 2)
    chunk_size = math.ceil(num_t / num_workers)
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for i in range(num_workers):
            t_start = i * chunk_size
            t_end = min((i + 1) * chunk_size, num_t)
            if t_start >= num_t: break
            
            futures.append(pool.submit(
                process_chunk,
                t_start, t_end, target_path, c_axis, extraction_plan,
                top_roi, bot_roi, base_name, output_dir_base,
                psf_map, z_step_um, args.debug, i
            ))
            
        for f in futures:
            f.result()
            
    print("All extraction and submission complete!")

if __name__ == "__main__":
    main()
