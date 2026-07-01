import os
import numpy as np
from pathlib import Path

# Create a fake input file
os.makedirs("/tmp/mock_data", exist_ok=True)
mock_file = Path("/tmp/mock_data/mock_cell.tif")
if not mock_file.exists():
    import tifffile
    store = np.zeros((1, 2, 50, 1024, 2048), dtype=np.uint16)
    tifffile.imwrite(mock_file, store, imagej=True)

from run_napari_opym import petakit_pipeline, _run_unified_job_chain
import napari

# Create mock objects for layers
class MockImageLayer:
    def __init__(self, path):
        class Source:
            def __init__(self, path):
                self.path = path
        self.source = Source(path)
        self.metadata = {}

class MockShapesLayer:
    def __init__(self):
        self.data = []

# Mock the widget attributes
class MockWidget:
    def __init__(self):
        class ValueObj:
            def __init__(self, val):
                self.value = val
        self.save_gfp = ValueObj(True)
        self.save_calcein_violet = ValueObj(False)
        self.save_mscarlet = ValueObj(True)
        self.save_cf647 = ValueObj(False)
        self.detected_z_step = 0.3

import run_napari_opym
run_napari_opym.pipeline_widget = MockWidget()

try:
    # We will trigger the pipeline directly
    print("Triggering petakit_pipeline...")
    petakit_pipeline(
        viewer=None,
        image_layer=MockImageLayer(str(mock_file)),
        shapes_layer=MockShapesLayer(),
    )
    print("petakit_pipeline finished without exceptions in the main thread!")
except Exception as e:
    import traceback
    traceback.print_exc()

