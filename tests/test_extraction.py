import time
import tifffile
import zarr
import numpy as np
path = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0_47.ome.tif"
store = tifffile.imread(path, aszarr=True)
z = zarr.open(store, mode='r')

st = time.time()
for c in [0, 1]:
    roi1 = np.array(z[0, c, :, 343:919, 710:1862])
    roi2 = np.array(z[0, c, :, 1376:1952, 627:1779])
print(f"4 sequential ROI reads took: {time.time()-st:.2f}s")
