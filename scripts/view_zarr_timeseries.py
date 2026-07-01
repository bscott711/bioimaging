import sys
import zarr
import dask.array as da
import napari
from pathlib import Path
import re
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_zarr_timeseries.py /path/to/Decon/")
        sys.exit(1)
        
    zarr_dir = Path(sys.argv[1])
    zarr_files = list(zarr_dir.glob("*.zarr"))
    
    if not zarr_files:
        print(f"❌ No .zarr files found in {zarr_dir}")
        sys.exit(1)
        
    # Group by channel
    channels = defaultdict(list)
    for z_path in zarr_files:
        # Extract Time (T) and Channel (C) from name, e.g. T0000_C0
        t_match = re.search(r'_T(\d+)', z_path.name)
        c_match = re.search(r'_C(\d+)', z_path.name)
        
        t = int(t_match.group(1)) if t_match else 0
        c = int(c_match.group(1)) if c_match else 0
        
        channels[c].append((t, z_path))
        
    print(f"🚀 Found {len(zarr_files)} Zarr stores across {len(channels)} channels.")
    
    viewer = napari.Viewer()
    
    colors = ['cyan', 'magenta', 'yellow', 'green']
    
    for i, (c, files) in enumerate(sorted(channels.items())):
        files.sort(key=lambda x: x[0]) # Sort by timepoint
        
        lazy_arrays = []
        for t, z_path in files:
            store = zarr.open(str(z_path), mode='r')
            # Handle standard Zarr vs OME-Zarr structure
            arr = store['0'] if '0' in store else store
            lazy_arrays.append(da.from_zarr(arr))
            
        # Stack all timepoints into a 4D array (T, Z, Y, X)
        stacked = da.stack(lazy_arrays, axis=0)
        print(f"   Channel {c} Stacked Shape: {stacked.shape}")
        
        # Add to Napari (blending="additive" allows multi-channel overlay)
        viewer.add_image(
            stacked, 
            name=f"Channel {c}", 
            colormap=colors[i % len(colors)],
            blending='additive'
        )
        
    print("🎬 Launching Napari...")
    napari.run()

if __name__ == "__main__":
    main()
