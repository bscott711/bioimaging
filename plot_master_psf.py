#!/usr/bin/env python3
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

master_path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/Master_PSF_Final_Strict.tif")
data = tifffile.imread(master_path)

while data.ndim > 3:
    data = data[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(np.max(data, axis=0), cmap='magma')
axes[0].set_title('Master PSF XY MIP')
axes[0].axis('off')

axes[1].imshow(np.max(data, axis=2), cmap='magma')
axes[1].set_title('Master PSF YZ MIP (showing shear)')
axes[1].axis('off')

out_png = master_path.with_name('Master_PSF_Final_Strict.png')
fig.savefig(out_png, bbox_inches='tight', dpi=150)
plt.close(fig)

print(f"Saved Master PSF visualization to {out_png}")
