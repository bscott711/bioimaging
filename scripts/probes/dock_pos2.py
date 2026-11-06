import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDockWidget, QLabel

viewer = napari.Viewer()
qt_win = viewer.window._qt_window

methods = [m for m in dir(qt_win) if 'DockWidget' in m]
print(methods)

