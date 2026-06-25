import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDockWidget, QLabel

viewer = napari.Viewer()
qt_win = viewer.window._qt_window

# Create a test dock widget
dock1 = QDockWidget("My Test Widget")
dock1.setWidget(QLabel("Hello"))
qt_win.addDockWidget(Qt.LeftDockWidgetArea, dock1)

# print current dock widgets
for dock in qt_win.findChildren(QDockWidget):
    print(dock.objectName(), dock.windowTitle())
