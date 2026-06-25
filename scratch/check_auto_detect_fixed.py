import zarr
import tifffile
import numpy as np
import scipy.ndimage
import skimage.measure

store = tifffile.imread("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0_47.ome.tif", aszarr=True)
z = zarr.open(store, mode='r')

t0_data = z[0, 0]
proj_axes = tuple(range(t0_data.ndim - 2))
max_proj = np.max(t0_data, axis=proj_axes)
max_y, max_x = max_proj.shape
half_y = max_y // 2

def _find_half_roi(image_half, offset_y=0):
    if np.max(image_half) < 300:
        return None
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

print("Top ROI:", _find_half_roi(max_proj[:half_y, :], 0))
print("Bot ROI:", _find_half_roi(max_proj[half_y:, :], half_y))
