import os
import napari
os.environ["QT_QPA_PLATFORM"] = "offscreen"
help(napari.Viewer().window.add_dock_widget)
