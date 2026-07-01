import sys
import zarr
import dask.array as da
import tifffile
from pathlib import Path
import re
from collections import defaultdict
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_mip_tif.py /path/to/Decon/")
        sys.exit(1)
        
    zarr_dir = Path(sys.argv[1])
    zarr_files = list(zarr_dir.glob("*.zarr"))
    
    if not zarr_files:
        print(f"❌ No .zarr files found in {zarr_dir}")
        sys.exit(1)
        
    channels = defaultdict(list)
    for z_path in zarr_files:
        t_match = re.search(r'_T(\d+)', z_path.name)
        c_match = re.search(r'_C(\d+)', z_path.name)
        t = int(t_match.group(1)) if t_match else 0
        c = int(c_match.group(1)) if c_match else 0
        channels[c].append((t, z_path))
        
    print(f"🚀 Found {len(zarr_files)} Zarr stores across {len(channels)} channels.")
    
    out_dir = zarr_dir.parent
    
    for c, files in sorted(channels.items()):
        files.sort(key=lambda x: x[0])
        
        lazy_arrays = []
        for t, z_path in files:
            store = zarr.open(str(z_path), mode='r')
            arr = store['0'] if '0' in store else store
            lazy_arrays.append(da.from_zarr(arr))
            
        stacked = da.stack(lazy_arrays, axis=0) # Shape: (T, Z, Y, X)
        
        print(f"   Calculating Maximum Intensity Projection (Z-axis) for Channel {c}...")
        # Max projection along Z axis (axis=1)
        mip = stacked.max(axis=1).compute() # Shape: (T, Y, X)
        
        out_path = out_dir / f"Decon_MIP_C{c}.tif"
        print(f"   Saving to {out_path} ...")
        tifffile.imwrite(out_path, mip, imagej=True, metadata={'axes': 'TYX'})
        
    print("✅ All channels exported!")
    print(f"You can now download the .tif files from {out_dir} to your local computer!")

if __name__ == "__main__":
    main()
