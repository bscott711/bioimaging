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
from opym.petakit import submit_pipeline_job, wait_for_job
from opym.utils import orient_zyx_for_dsr

target_path = Path("/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0.ome.tif")
psf_path = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif"
# This PSF file predates dz_psf metadata tagging (see psf_tools/extract_bead_psf.py)
# and lives on shared scratch, so it's passed explicitly here rather than
# read from the file. Bead stack was acquired at 0.1um/step (per its source
# folder naming, e.g. ".../0p1micron/...").
psf_dz_um = 0.1
# _v2: the pipeline fix (dz_psf resampling, axis-order correction, GPU
# concurrency lock) landed after the original AngleSweep/ results were
# collected, so those are stale/pre-fix. Writing to a new directory rather
# than overwriting them.
output_dir_base = Path("/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0/Decon/AngleSweep_v2")

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
    cropped = orient_zyx_for_dsr(cropped)
        
    angles = range(5, 90, 5) # 5 to 85

    # Submitted sequentially (submit -> wait -> next) rather than all at once.
    # The server now enforces its own one-job-per-GPU lock, but doing it here
    # too gives an explicit per-angle pass/fail record instead of silently
    # missing directories if something still goes wrong (e.g. GPU OOM from
    # some other job on the node) -- a burst submission previously dropped
    # 2 of 17 angles to OOM with no obvious sign in compare_angles_napari.py.
    succeeded = []
    failed = []
    for angle in angles:
        angle = float(angle)
        print(f"Submitting job for angle {angle}...")

        shm_path = Path(f"/dev/shm/opym_jobs/sweep_{angle}.zarr")
        shm_path.parent.mkdir(exist_ok=True, parents=True)
        zarr.save(shm_path, cropped)

        # run_gpu_pipeline outputs .zarr
        output_file = output_dir_base / f"Angle_{angle}" / "DSR" / f"sweep_{angle}.zarr"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        job_path = submit_pipeline_job(
            output_file=output_file,
            shm_path=shm_path,
            psf_paths=[psf_path],
            z_step_um=z_step_um,
            sheet_angle_deg=angle,
            dz_psf=psf_dz_um,
            save_zarr=True,
            queue_dir=Path("/dev/shm/petakit_jobs/queue"),
        )

        ok = wait_for_job(job_path, poll_interval=2)
        if ok:
            succeeded.append(angle)
        else:
            failed.append(angle)
            print(f"WARNING: angle {angle} FAILED -- check /dev/shm/petakit_jobs/failed/")

    print(f"\nSweep complete: {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed angles (missing/stale output, do not trust these in compare_angles_napari.py): {failed}")
    print(f"Succeeded angles: {succeeded}")

if __name__ == '__main__':
    main()
