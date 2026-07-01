import tifffile
import numpy as np
from scipy.optimize import curve_fit
import scipy.ndimage as ndi

def gaussian_1d(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

# Load the raw cropped image
img_path = '/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0_test/cell_MMStack_Pos0_test_GFP_T0000.tif'
img = tifffile.imread(img_path)
if img.ndim > 3: img = img[0]

print("Image shape:", img.shape)
Z_dim, Y_dim, X_dim = img.shape

# The image is skewed. To find the true Z width, we should look at edges.
# Or better, we can just find some bright isolated spots.
# Let's apply a Laplacian or high-pass filter to find sharp puncta.
filtered = ndi.gaussian_laplace(img.astype(np.float32), sigma=1.0)
# Find minimum of laplacian (sharpest bright spots)
min_val = np.min(filtered)
threshold = min_val * 0.5

peaks = np.argwhere(filtered < threshold)
print(f"Found {len(peaks)} sharp spots. Measuring the top 5...")

sigmas_z = []
sigmas_y = []
sigmas_x = []

for idx, (z, y, x) in enumerate(peaks[:20]):
    try:
        # Extract 1D profiles (in pixel space)
        # Note: In skewed space, Z profile is slanted, so its width is wider.
        pad = 7
        if z < pad or z > Z_dim-pad or y < pad or y > Y_dim-pad or x < pad or x > X_dim-pad:
            continue
            
        z_prof = img[z-pad:z+pad+1, y, x]
        y_prof = img[z, y-pad:y+pad+1, x]
        x_prof = img[z, y, x-pad:x+pad+1]
        
        pz, _ = curve_fit(gaussian_1d, np.arange(len(z_prof)), z_prof, p0=[np.max(z_prof)-np.min(z_prof), pad, 2.0, np.min(z_prof)])
        py, _ = curve_fit(gaussian_1d, np.arange(len(y_prof)), y_prof, p0=[np.max(y_prof)-np.min(y_prof), pad, 2.0, np.min(y_prof)])
        px, _ = curve_fit(gaussian_1d, np.arange(len(x_prof)), x_prof, p0=[np.max(x_prof)-np.min(x_prof), pad, 2.0, np.min(x_prof)])
        
        sz, sy, sx = abs(pz[2]), abs(py[2]), abs(px[2])
        if sz < 10 and sy < 10 and sx < 10: # Reasonable
            sigmas_z.append(sz)
            sigmas_y.append(sy)
            sigmas_x.append(sx)
    except:
        pass

if len(sigmas_z) > 0:
    med_sz = np.median(sigmas_z)
    med_sy = np.median(sigmas_y)
    med_sx = np.median(sigmas_x)
    print(f"Median Sigmas (pixels): Z={med_sz:.2f}, Y={med_sy:.2f}, X={med_sx:.2f}")
    # Convert to physical microns.
    # Note: Z step in data is 0.3, X/Y is 0.136
    phys_z = med_sz * 0.3
    phys_y = med_sy * 0.136
    phys_x = med_sx * 0.136
    
    # Correct Z for skew (divide by cos(30) if we measured straight down the slanted volume)
    # Z_true = Z_apparent * cos(30)
    theta = np.radians(30)
    true_z = phys_z * np.cos(theta)
    
    print(f"Physical Unskewed Sigmas (microns): Z={true_z:.2f}, Y={phys_y:.2f}, X={phys_x:.2f}")
    print(f"Physical Unskewed FWHM (microns): Z={true_z*2.355:.2f}, Y={phys_y*2.355:.2f}, X={phys_x*2.355:.2f}")
else:
    print("Could not measure spots.")

