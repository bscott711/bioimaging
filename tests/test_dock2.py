import os
import sys
import runpy
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import napari
original_run = napari.run
def fake_run():
    import __main__
    viewer = __main__.viewer
    print("Dock widgets:")
    for name, widget in viewer.window.dock_widgets.items():
        print(f" - {name}: {widget}")
napari.run = fake_run

sys.argv = ["run_napari_opym.py"]
runpy.run_path("run_napari_opym.py", run_name="__main__")
