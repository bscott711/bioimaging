import os
import sys
import runpy
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import napari
original_run = napari.run
def fake_run():
    print("Napari run called!")
    import __main__
    viewer = __main__.viewer
    print("Dock widgets:", viewer.window._dock_widgets)
napari.run = fake_run

sys.argv = ["run_napari_opym.py"]
runpy.run_path("run_napari_opym.py", run_name="__main__")
