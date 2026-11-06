# Probes

One-off scripts that used to sit in `tests/` as `test_*.py` and ran at pytest collection. They are manual:
run one by hand, knowing what it touches. They may write to `/mmfs2` or `/tmp`, and must never be run by pytest
(`pyproject.toml` limits `testpaths` to `tests/` and skips `scripts/`).

| probe | what it does |
|---|---|
| `conver_to_zarr.py` | Deletes and rewrites `/mmfs2/scratch/SDSMT.LOCAL/bscott/benchmark_zarr_{uncompressed,blosc}.zarr` (8 timepoints, about 32 GB) from a raw OME-TIFF on `/mmfs2`, timing each write. |
| `extraction.py` | Times four ROI reads from a 4.3 GB raw OME-TIFF on `/mmfs2` through tifffile's zarr view (read only). |
| `logic.py` | Prints the fluorophores `parse_name` finds in one dataset folder name (pure regex, touches no files). |
| `pipeline_headless.py` | Writes a 419 MB `/tmp/mock_data/mock_cell.tif`, imports `run_napari_opym` and calls `petakit_pipeline` with mock layers. Past ROI detection it could queue tickets in the production jobs dir. |
| `main.py` | Imports napari, replaces `napari.run`, and runs `run_napari_opym.py` as `__main__` (opens a napari Viewer). |
| `dock.py` | Same as `main.py`, then prints `viewer.window._dock_widgets` (opens a napari Viewer). |
| `dock2.py` | Same as `dock.py`, but lists `viewer.window.dock_widgets` by name (opens a napari Viewer). |
| `start.py` | Imports `run_napari_opym` and nothing else. |
| `widget.py` | Imports `run_napari_opym` and builds the `load_lazy_ome_tiff` widget. |
| `move.py` | Opens a napari Viewer, adds a dock widget and calls `splitDockWidget` on the layer controls. |
| `split.py` | Opens a napari Viewer, adds a dock widget and looks up the layer controls and layer list docks. |
| `help.py` | Opens a napari Viewer and prints `help()` for `add_dock_widget`. |
| `dock_pos.py` | Opens a napari Viewer, adds a Qt dock widget on the left and lists the dock widgets. |
| `dock_pos2.py` | Opens a napari Viewer and lists the `DockWidget` methods of its Qt window. |
| `extraction_plan_smoke.py` | Prints `get_extraction_plan` for one raw OME-TIFF on `/mmfs2`, under `__main__` only (read only). |

Other probes that were `test_*.py` outside `tests/`: `scripts/decon_probe.py` (prints "Starting..." and nothing else) and
`scratch/mp_probe.py` (two processes sharing a `multiprocessing.Value`, prints which worker got which counter).
