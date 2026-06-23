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
    
    def _find_half_roi(image_half, offset_y=0):
        smoothed = scipy.ndimage.gaussian_filter(image_half, sigma=5)
        thresh = np.mean(smoothed) + 2 * np.std(smoothed)
        mask = smoothed > thresh
        labels = skimage.measure.label(mask)
        props = skimage.measure.regionprops(labels)
        if not props: return None
        
        min_row, min_col, max_row, max_col = image_half.shape[0], image_half.shape[1], 0, 0
        found = False
        for p in props:
            if p.area > 500:
                r0, c0, r1, c1 = p.bbox
                min_row, min_col = min(min_row, r0), min(min_col, c0)
                max_row, max_col = max(max_row, r1), max(max_col, c1)
                found = True
                
        if not found: return None
        
        return {
            "ymin": min_row + offset_y,
            "ymax": max_row + offset_y,
            "xmin": min_col,
            "xmax": max_col
        }

    top_roi_dict = _find_half_roi(max_proj[:half_y, :], 0)
    bot_roi_dict = _find_half_roi(max_proj[half_y:, :], half_y)
    
    valid_rois = [r for r in [top_roi_dict, bot_roi_dict] if r is not None]
    rois_out = []
    
    if valid_rois:
        detected_max_h = max(r["ymax"] - r["ymin"] for r in valid_rois)
        detected_max_w = max(r["xmax"] - r["xmin"] for r in valid_rois)
        
        EXPECTED_H = 576
        EXPECTED_W = 1152
            
        max_h = max(detected_max_h, EXPECTED_H)
        max_w = max(detected_max_w, EXPECTED_W)
        
        if master_roi_path:
            try:
                with open(master_roi_path, 'w') as f:
                    json.dump({"max_h": max_h, "max_w": max_w}, f)
            except: pass
            
        for r_dict in [top_roi_dict, bot_roi_dict]:
            if r_dict is None:
                rois_out.append(None)
                continue
                
            y_center = (r_dict["ymin"] + r_dict["ymax"]) // 2
            x_center = (r_dict["xmin"] + r_dict["xmax"]) // 2
            
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

def main():
    parser = argparse.ArgumentParser(description="End-to-End GPU Pipeline CLI")
    parser.add_argument("data_path", type=str, help="Path to the directory containing the ome.tif file")
    parser.add_argument("--psf", type=str, default="/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif", help="Path to PSF")
    args = parser.parse_args()

    data_dir = Path(args.data_path).resolve()
    
    # Find the ome.tif file
    ome_tifs = list(data_dir.glob("*.ome.tif"))
    if not ome_tifs:
        print(f"No .ome.tif found in {data_dir}")
        return
        
    target_path = ome_tifs[0]
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
        
    print(f"Detected Zarr shape: {z.shape}. T={num_t}, C={num_c}. Extracting all timepoints...")
    
    def process_chunk_wrapper(t, c, is_top, roi):
        roi_name = "top" if is_top else "bot"
        shm_path = Path(f"/dev/shm/opym_jobs/{target_path.stem}_T{t:04d}_C{c}_{roi_name}.tif")
        output_dir = data_dir / ("Top" if is_top else "Bot")
        
        # Throttle so we don't fill up /dev/shm
        shm_dir = Path("/dev/shm/opym_jobs")
        while shm_dir.exists() and len(list(shm_dir.glob("*.tif"))) > 100:
            time.sleep(1)
            
        print(f"[{t}:{c}:{roi_name}] Extracting ROI from Zarr...")
        st = time.time()
        
        # Correctly slice the dimensions to yield (Z, Y, X)
        if z.ndim == 5:
            if c_axis == 1:
                cropped = z[t, c, :, roi[0], roi[1]]
            else:
                cropped = z[t, :, c, roi[0], roi[1]]
        else:
            print("Unexpected dimensions. Skipping.")
            return

        print(f"[{t}:{c}:{roi_name}] Extracted {cropped.shape} in {time.time()-st:.2f}s")
        
        shm_path.parent.mkdir(exist_ok=True, parents=True)
        cropped_mem = np.array(cropped)
        
        print(f"[{t}:{c}:{roi_name}] Writing to {shm_path}...")
        st2 = time.time()
        tifffile.imwrite(shm_path, cropped_mem, imagej=True)
        print(f"[{t}:{c}:{roi_name}] Written in {time.time()-st2:.2f}s")
        
        output_dir = data_dir.parent / "cell_1_FAST" / ("Top" if is_top else "Bot")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / shm_path.name
        
        submit_pipeline_job(
            output_file=output_file,
            shm_path=shm_path,
            psf_paths=[args.psf],
            z_step_um=z_step_um,
            iterations=10,
        )
        print(f"[{t}:{c}:{roi_name}] Job Submitted!")
        
    print("Starting sequential extraction for maximum GPFS throughput...")
    for t in range(num_t):
        for c in range(num_c):
            if top_roi:
                process_chunk_wrapper(t, c, True, top_roi)
            if bot_roi:
                process_chunk_wrapper(t, c, False, bot_roi)
                
    print("All extraction and submission complete!")

if __name__ == "__main__":
    main()
