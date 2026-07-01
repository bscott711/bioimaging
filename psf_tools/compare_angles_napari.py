import napari
import tifffile
import zarr
import numpy as np
from pathlib import Path
from magicgui import magicgui
from scipy.ndimage import center_of_mass

sweep_dir_rotate = Path('/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0/Decon/AngleSweep')

def get_angles(*args, **kwargs):
    angles = set()
    for d in sweep_dir_rotate.glob('Angle_*'):
        try: angles.add(float(d.name.split('_')[1]))
        except: pass
    return sorted(list(angles)) if angles else [28.0]

viewer = napari.Viewer(title="Deskew Angle Comparator")

@magicgui(
    call_button="Load Overlay",
    angle_1={"choices": get_angles},
    angle_2={"choices": get_angles},
    angle_3={"choices": get_angles},
)
def overlay_angles(angle_1: float, angle_2: float, angle_3: float):
    # Paths to DSR files
    dir_1 = sweep_dir_rotate / f'Angle_{angle_1}' / 'DSR'
    dir_2 = sweep_dir_rotate / f'Angle_{angle_2}' / 'DSR'
    dir_3 = sweep_dir_rotate / f'Angle_{angle_3}' / 'DSR'
    
    files_1 = list(dir_1.glob('*.zarr'))
    files_2 = list(dir_2.glob('*.zarr'))
    files_3 = list(dir_3.glob('*.zarr'))
    
    if not files_1 or not files_2 or not files_3:
        print(f"Error: Missing DSR data for one of the selected angles.")
        return
        
    print(f"Loading Angle {angle_1}...")
    data_1 = zarr.open(files_1[0], mode='r')[:]
    print(f"Loading Angle {angle_2}...")
    data_2 = zarr.open(files_2[0], mode='r')[:]
    print(f"Loading Angle {angle_3}...")
    data_3 = zarr.open(files_3[0], mode='r')[:]
    
    # Clear existing layers
    viewer.layers.clear()
    
    # Fast Center of Mass calculation using 1D marginals
    print("Calculating alignments...")
    def fast_com(data):
        z_marg = np.sum(data, axis=(1,2))
        y_marg = np.sum(data, axis=(0,2))
        x_marg = np.sum(data, axis=(0,1))
        
        z_com = np.sum(z_marg * np.arange(len(z_marg))) / np.sum(z_marg)
        y_com = np.sum(y_marg * np.arange(len(y_marg))) / np.sum(y_marg)
        x_com = np.sum(x_marg * np.arange(len(x_marg))) / np.sum(x_marg)
        return np.array([z_com, y_com, x_com])

    com_1 = fast_com(data_1)
    com_2 = fast_com(data_2)
    com_3 = fast_com(data_3)
    
    # Add layer 1 (Magenta)
    layer_1 = viewer.add_image(
        data_1, 
        name=f"Angle {angle_1}", 
        colormap="magenta", 
        blending="additive",
        opacity=1.0
    )
    
    # Add layer 2 (Green)
    layer_2 = viewer.add_image(
        data_2, 
        name=f"Angle {angle_2}", 
        colormap="green", 
        blending="additive",
        opacity=1.0
    )
    
    # Add layer 3 (Cyan)
    layer_3 = viewer.add_image(
        data_3, 
        name=f"Angle {angle_3}", 
        colormap="cyan", 
        blending="additive",
        opacity=1.0
    )
    
    # Shift layer 2 and 3 so their center of mass matches layer 1
    layer_2.translate = com_1 - com_2
    layer_3.translate = com_1 - com_3
    
    viewer.reset_view()
    print("Done!")

@magicgui(call_button="Refresh Available Angles")
def refresh_angles():
    new_angles = get_angles()
    overlay_angles.angle_1.choices = new_angles
    overlay_angles.angle_2.choices = new_angles
    overlay_angles.angle_3.choices = new_angles
    print(f"Angles refreshed: {len(new_angles)} angles available.")

# Add widgets
viewer.window.add_dock_widget(refresh_angles, name="Refresh", area='right')
viewer.window.add_dock_widget(overlay_angles, name="Comparator", area='right')

if __name__ == '__main__':
    napari.run()
