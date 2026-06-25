import zarr
import tifffile
import numpy as np
store = tifffile.imread("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0_47.ome.tif", aszarr=True)
z = zarr.open(store, mode='r')
print(f"Shape: {z.shape}")
c0 = z[0,0]
c1 = z[0,1]
half_y = c0.shape[1] // 2
print(f"C0 Top Max: {np.max(c0[:, :half_y, :])}")
print(f"C0 Bot Max: {np.max(c0[:, half_y:, :])}")
print(f"C1 Top Max: {np.max(c1[:, :half_y, :])}")
print(f"C1 Bot Max: {np.max(c1[:, half_y:, :])}")
