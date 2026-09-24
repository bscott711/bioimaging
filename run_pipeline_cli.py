#!/usr/bin/env python3
import argparse
import time
import json
import numpy as np
import tifffile
import zarr
from pathlib import Path
import math
import multiprocessing as mp

import sys
# Make sure we can import opym from local projects
sys.path.append(str(Path.home() / "projects" / "opym_local" / "src"))
from opym.petakit import submit_pipeline_batch_job
from opym.consolidate import consolidate_to_ome_zarr
from opym.utils import orient_zyx_for_dsr
from opym.roi_detect import auto_detect_rois, compute_reference_projection
from psf_tools.extraction_plan import get_extraction_plan

# Max (timepoint, channel) frames bundled into one pipeline_batch ticket.
# Frames are grouped by PSF (laser/channel) first, then batched across
# consecutive timepoints up to this size -- see the performance plan's
# Phase 2. Deliberately a named, tunable constant: raised from the initial
# default of 8 to 30 to measure whether amortizing per-ticket overhead
# further is worth the larger failure blast-radius and coarser progress
# granularity (see Phase 2.3's real-data timing).
MAX_BATCH_SIZE = 30

def get_z_step(target_path: Path):
    acq_settings = target_path.parent / "AcqSettings.txt"
    if acq_settings.exists():
        with open(acq_settings, 'r') as f:
            d = json.load(f)
            return float(d.get("stepSizeUm", 0.3))
    return 0.3

def process_chunk(t_start, t_end, target_path, c_axis, extraction_plan, top_roi, bot_roi, base_name, output_dir_base, psf_map, z_step_um, psf_dz_um, debug, worker_id):
    """
    Process a contiguous chunk of timepoints directly from GPFS.
    Opens the multi-file Zarr store ONCE (paying a one-time metadata tax),
    then streams sequentially through the assigned range.
    """
    import tifffile, zarr, time, numpy as np
    from pathlib import Path
    from opym.petakit import submit_pipeline_batch_job
    from opym.utils import orient_zyx_for_dsr

    # Note: tifffile memory-maps by default when possible. The boundary delays
    # are due to GPFS file open/close token overhead when crossing 4GB splits.
    store = tifffile.imread(str(target_path), aszarr=True)
    z = zarr.open(store, mode='r')

    print(f"[Worker {worker_id:02d}] Initialized. Processing T={t_start} to {t_end-1}")

    # Group by (laser_name == PSF) first, then batch consecutive timepoints
    # within each group up to MAX_BATCH_SIZE -- see the performance plan's
    # Phase 2. Per-frame extract/orient/stage-write logic below is
    # unchanged; only dispatch granularity (one ticket per MAX_BATCH_SIZE
    # frames instead of one per frame) differs from before.
    for c, is_top, laser_name, output_c in extraction_plan:
        roi = top_roi if is_top else bot_roi
        if not roi: continue

        psf_path = psf_map.get(laser_name, None)
        batch_items = []

        def _flush_batch():
            if not batch_items:
                return
            submit_pipeline_batch_job(
                items=[{"shm_path": str(sp), "output_file": str(of)} for sp, of in batch_items],
                psf_paths=[psf_path] if psf_path else None,
                z_step_um=z_step_um, dz_psf=psf_dz_um if psf_path else None,
                debug=debug,  # let the MATLAB-side default (25 iters for RLMethod='simple') apply
                ticket_label=f"{base_name}_C{output_c}",
            )
            batch_items.clear()

        for t in range(t_start, t_end):
            shm_path = Path(f"/dev/shm/opym_jobs/{base_name}_T{t:04d}_C{output_c}.zarr")
            st = time.time()

            # BULK SINGLE-READ
            if z.ndim == 5:
                if c_axis == 1: cropped = np.array(z[t, c, :, roi[0], roi[1]])
                else: cropped = np.array(z[t, :, c, roi[0], roi[1]])
            else: continue

            extract_time = time.time() - st

            shm_path.parent.mkdir(exist_ok=True, parents=True)
            oriented = orient_zyx_for_dsr(cropped)
            # This is a write-once/read-once-wholesale staging file -- MATLAB's
            # readzarr() always reads the whole array back (run_gpu_pipeline.m).
            # Store it as a single chunk instead of zarr's default multi-chunk
            # heuristic, to skip unneeded chunk-coordination/decompression
            # overhead for data that's never partially accessed.
            # NOTE: zarr.save() silently routes to save_group() (not
            # save_array()) whenever *any* kwarg is passed -- even chunks= --
            # which would nest this under an arr_0 group entry instead of
            # writing a plain array. Must call save_array() directly.
            zarr.save_array(shm_path, oriented, chunks=oriented.shape)

            output_file = output_dir_base / shm_path.name
            batch_items.append((shm_path, output_file))

            print(f"[Worker {worker_id:02d}] T={t:03d} C={output_c} | Extract: {extract_time:.2f}s (staged for batch)")

            if len(batch_items) >= MAX_BATCH_SIZE:
                _flush_batch()

        _flush_batch()  # tail flush for whatever's left under MAX_BATCH_SIZE



def main():
    parser = argparse.ArgumentParser(description="End-to-End GPU Pipeline CLI")
    parser.add_argument("data_path", type=str, help="Path to the directory containing the ome.tif file")
    parser.add_argument("--psf", type=str, default="/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif", help="Path to PSF")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and profiling")
    parser.add_argument("--t0-only", action="store_true", help="Process only the first timepoint (T=0) for rapid testing")
    parser.add_argument("--consolidate", action="store_true",
                        help="Skip pipeline; consolidate existing Decon/*.zarr files into a single OME-Zarr")
    parser.add_argument("--xy-pixel-um", type=float, default=0.136,
                        help="XY pixel size in µm for OME-Zarr metadata (default: 0.136)")
    parser.add_argument("--psf-dz-um", type=float, default=0.1,
                        help="Z-step (µm) the PSF file was acquired at, used to resample the PSF "
                             "to this dataset's z-geometry before deconvolution (default: 0.1)")
    parser.add_argument("--channel-names", type=str, nargs="+", default=[],
                        help="Channel labels for OME-Zarr metadata, e.g. --channel-names GFP mScarlet")
    args = parser.parse_args()

    data_dir = Path(args.data_path).resolve()

    if args.consolidate:
        decon_dir = data_dir / "Decon"
        import re as _re
        ome_tifs = list(data_dir.glob("*.ome.tif"))
        if ome_tifs:
            target_path = min(ome_tifs, key=lambda p: len(p.name))
            base_name = _re.sub(r'_\d+\.ome$', '', target_path.stem)
            z_step_um = get_z_step(target_path)
            t_interval_s = 1.0
            acq = target_path.parent / "AcqSettings.txt"
            if acq.exists():
                with open(acq) as f:
                    d = json.load(f)
                    t_interval_s = float(d.get("timepointInterval", 1.0))
        else:
            # Infer base name from first zarr in Decon/
            sample = next((p for p in decon_dir.iterdir() if p.suffix == '.zarr' and '_T' in p.name), None)
            if not sample:
                print("Cannot infer base name — no zarrs found in Decon/")
                return
            base_name = sample.name.split('_T')[0]
            z_step_um = 0.3
            t_interval_s = 1.0

        consolidate_to_ome_zarr(
            decon_dir=decon_dir,
            base_name=base_name,
            z_step_um=z_step_um,
            xy_pixel_um=args.xy_pixel_um,
            t_interval_s=t_interval_s,
            channel_names=args.channel_names,
        )
        return

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
    
    max_proj = compute_reference_projection(z, timepoint=z.shape[0] // 2)
    top_roi, bot_roi = auto_detect_rois(max_proj, master_roi_path=master_roi_path)
    
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

    # Build the complete list of zarrs that MATLAB will produce
    expected_zarrs = []
    for t in range(num_t):
        for c, is_top, laser_name, output_c in extraction_plan:
            roi = top_roi if is_top else bot_roi
            if roi:
                expected_zarrs.append(f"{base_name}_T{t:04d}_C{output_c}.zarr")

    # Read timepoint interval from AcqSettings.txt if available
    t_interval_s = 1.0
    acq = target_path.parent / "AcqSettings.txt"
    if acq.exists():
        with open(acq) as f:
            acq_data = json.load(f)
            t_interval_s = float(acq_data.get("timepointInterval", 1.0))

    # Write sidecar so opym-serve can consolidate when MATLAB finishes
    output_dir_base.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir_base / ".opym_consolidate.json"
    with open(sidecar_path, 'w') as f:
        json.dump({
            "base_name": base_name,
            "expected_zarrs": expected_zarrs,
            "xy_pixel_um": args.xy_pixel_um,
            "z_step_um": z_step_um,
            "t_interval_s": t_interval_s,
            "channel_names": args.channel_names,
        }, f, indent=2)

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
                psf_map, z_step_um, args.psf_dz_um, args.debug, i
            ))

        for f in futures:
            f.result()

    print("All extraction and submission complete!")
    print("🔗 opym-serve will consolidate into OME-Zarr once MATLAB finishes")

if __name__ == "__main__":
    main()
