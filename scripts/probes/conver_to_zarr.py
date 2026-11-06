import tifffile
import zarr
import numpy as np
from pathlib import Path
import time
import shutil

# --- CONFIGURATION ---
src = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/20260402_py_FLM_2XFyve_mSca_mem_NG/cell_2/cell_MMStack_Pos0.ome.tif"
dst_uncomp = "/mmfs2/scratch/SDSMT.LOCAL/bscott/benchmark_zarr_uncompressed.zarr"
dst_blosc = "/mmfs2/scratch/SDSMT.LOCAL/bscott/benchmark_zarr_blosc.zarr"

num_t = 8
# PERFECT CHUNK SHAPE: 1 Timepoint, All Z, 1 Channel, Full Y, Full X
# Avoids the 2GB Blosc limit and perfectly matches the GPU read pattern.
chunks = (1, 175, 1, 2400, 2400) 

# Clean up previous runs
if Path(dst_uncomp).exists(): shutil.rmtree(dst_uncomp)
if Path(dst_blosc).exists(): shutil.rmtree(dst_blosc)

# --- PHASE 1: OPEN AND CONVERT ---
print("Opening OME-TIFF as Zarr (Zero I/O overhead)...")
store = tifffile.imread(src, aszarr=True)
z_in = zarr.open(store, mode='r')

shape_8t = (num_t,) + z_in.shape[1:]

print(f"\n[1/4] Writing UNCOMPRESSED Zarr ({num_t} timepoints)...")
st = time.time()
z_uncomp = zarr.open(dst_uncomp, mode='w', shape=shape_8t, dtype=z_in.dtype, 
                     chunks=chunks, compressor=None)
for t in range(num_t):
    z_uncomp[t] = z_in[t] 
print(f"Written in {time.time()-st:.1f}s")

print(f"\n[2/4] Writing BLOSC-LZ4 Zarr ({num_t} timepoints)...")
st = time.time()
z_blosc = zarr.open(dst_blosc, mode='w', shape=shape_8t, dtype=z_in.dtype,
                    chunks=chunks, 
                    compressor=zarr.Blosc(cname='lz4', clevel=1, shuffle=zarr.Blosc.SHUFFLE))
for t in range(num_t):
    z_blosc[t] = z_in[t]
print(f"Written in {time.time()-st:.1f}s")

# --- PHASE 2: FILE SIZES ---
def get_dir_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / 1e9

print(f"\n[3/4] Storage Footprint:")
print(f"Uncompressed Size: {get_dir_size(dst_uncomp):.2f} GB")
print(f"Blosc-LZ4 Size:    {get_dir_size(dst_blosc):.2f} GB")

# --- PHASE 3: READ BENCHMARK ---
roi_y = slice(219, 795)
roi_x = slice(647, 1799)

print(f"\n[4/4] Benchmarking Read Speed (Cropped ROI)...")
for label, path in [("Uncompressed", dst_uncomp), ("Blosc-LZ4", dst_blosc)]:
    z = zarr.open(path, mode='r')
    
    # Warm up the OS page cache
    _ = z[0, 0, :, roi_y, roi_x]
    
    times = []
    for t in range(num_t):
        st = time.time()
        # Note: we read z[t, c, :, y, x] to match the exact chunk shape!
        cropped = np.array(z[t, 0, :, roi_y, roi_x]) 
        times.append(time.time() - st)
    
    avg = np.mean(times)
    print(f"{label:12} | Avg read+crop = {avg:.3f}s | Shape = {cropped.shape}")