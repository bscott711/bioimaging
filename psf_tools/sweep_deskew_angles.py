#!/usr/bin/env python3
import tifffile
import zarr
import time
import numpy as np
from pathlib import Path
import sys

# Add bioimaging so we can import auto_detect_rois and get_z_step
sys.path.append("/home/SDSMT.LOCAL/bscott/projects/bioimaging")
from run_pipeline_cli import auto_detect_rois, get_z_step

# Add opym_local to path to import petakit if needed
sys.path.append("/home/SDSMT.LOCAL/bscott/projects/opym_local/src")
from opym.petakit import submit_pipeline_job

target_path = Path("/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0.ome.tif")
psf_path = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"
output_dir_base = Path("/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0/Decon/AngleSweep")

def main():
    print(f"Reading target: {target_path}")
    store = tifffile.imread(str(target_path), aszarr=True)
    z = zarr.open(store, mode='r')
    
    z_step_um = get_z_step(target_path)
    if z_step_um is None: z_step_um = 0.3
    
    top_roi, bot_roi = auto_detect_rois(z)
    roi = top_roi if top_roi else bot_roi
    if not roi:
        roi = (slice(None), slice(None))
        
    t, c = 0, 0
    c_axis = 1 if z.shape[1] < z.shape[2] else 2
    
    print(f"Extracting T={t}, C={c} with ROI={roi}...")
    if z.ndim == 5:
        if c_axis == 1:
            cropped = np.array(z[t, c, :, roi[0], roi[1]])
        else:
            cropped = np.array(z[t, :, c, roi[0], roi[1]])
    else:
        raise ValueError("Expected 5D data")
        
    angles = range(5, 90, 5) # 5 to 85
    
    for angle in angles:
        angle = float(angle)
        print(f"Submitting job for angle {angle}...")
        
        shm_path = Path(f"/dev/shm/opym_jobs/sweep_{angle}.zarr")
        shm_path.parent.mkdir(exist_ok=True, parents=True)
        zarr.save(shm_path, cropped)
        
        # run_gpu_pipeline outputs .zarr
        output_file = output_dir_base / f"Angle_{angle}" / "DSR" / f"sweep_{angle}.zarr"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        submit_pipeline_job(
            output_file=output_file,
            shm_path=shm_path,
            psf_paths=[psf_path],
            z_step_um=z_step_um,
            sheet_angle_deg=angle,
            save_zarr=True,
            queue_dir=Path("/dev/shm/petakit_jobs/queue")
        )
        
    print("All jobs submitted to /dev/shm/petakit_jobs/queue/")
    print("Wait for opym-serve to finish processing them.")

if __name__ == '__main__':
    main()
