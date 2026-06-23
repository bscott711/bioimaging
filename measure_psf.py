import tifffile
import numpy as np
from scipy.optimize import curve_fit

def gaussian_1d(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

psf = tifffile.imread('/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/Master_PSF_Final_Strict.tif')
while psf.ndim > 3: psf = psf[0]

z_idx, y_idx, x_idx = np.unravel_index(np.argmax(psf), psf.shape)
print(f"Center at Z={z_idx}, Y={y_idx}, X={x_idx}")

z_prof = psf[:, y_idx, x_idx]
y_prof = psf[z_idx, :, x_idx]
x_prof = psf[z_idx, y_idx, :]

try:
    popt_z, _ = curve_fit(gaussian_1d, np.arange(len(z_prof)), z_prof, p0=[np.max(z_prof), z_idx, 2.0, 0])
    popt_y, _ = curve_fit(gaussian_1d, np.arange(len(y_prof)), y_prof, p0=[np.max(y_prof), y_idx, 2.0, 0])
    popt_x, _ = curve_fit(gaussian_1d, np.arange(len(x_prof)), x_prof, p0=[np.max(x_prof), x_idx, 2.0, 0])

    print(f"Master PSF Sigmas (pixels): Z={abs(popt_z[2]):.2f}, Y={abs(popt_y[2]):.2f}, X={abs(popt_x[2]):.2f}")
    print(f"Master PSF Sigmas (microns): Z={abs(popt_z[2])*0.1:.2f}, Y={abs(popt_y[2])*0.136:.2f}, X={abs(popt_x[2])*0.136:.2f}")
except Exception as e:
    print(e)
