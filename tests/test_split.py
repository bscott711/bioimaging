import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDockWidget, QLabel

viewer = napari.Viewer()
qt_win = viewer.window._qt_window

dock1 = viewer.window.add_dock_widget(QLabel("1"), name="Dock 1", area="left")

layer_controls = [d for d in qt_win.findChildren(QDockWidget) if d.objectName() == 'layer controls'][0]
layer_list = [d for d in qt_win.findChildren(QDockWidget) if d.objectName() == 'layer list'][0]

print("Successfully grabbed layer_controls and layer_list")
# Actually there's no easy way to print visual order without X11.
# Let's see if Qt splitDockWidget places the FIRST arg ABOVE the SECOND arg, or what.
