import os
import sys
import runpy
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.argv = ["run_napari_opym.py"]

# Mock napari.run so it doesn't block forever
import napari
original_run = napari.run
napari.run = lambda: print("Napari run called successfully!")

runpy.run_path("run_napari_opym.py", run_name="__main__")
