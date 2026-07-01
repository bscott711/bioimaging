import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDockWidget, QLabel

viewer = napari.Viewer()
qt_win = viewer.window._qt_window

# Add custom widget
my_dock = viewer.window.add_dock_widget(QLabel("Hello"), name="My Widget", area="left")

# Find layer controls
layer_controls = [d for d in qt_win.findChildren(QDockWidget) if d.objectName() == 'layer controls']
if layer_controls:
    # my_dock goes first, then layer_controls (so my_dock is on top)
    qt_win.splitDockWidget(my_dock, layer_controls[0], Qt.Vertical)
    print("Successfully called splitDockWidget")
