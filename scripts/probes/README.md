# Probes

One-off scripts that used to sit in `tests/` as `test_*.py` and ran at pytest collection. They are manual:
run one by hand, knowing what it touches. They may write to `/mmfs2` or `/tmp`, and must never be run by pytest
(`pyproject.toml` limits `testpaths` to `tests/` and skips `scripts/`).

| probe | what it does |
|---|---|
| `conver_to_zarr.py` | Deletes and rewrites `/mmfs2/scratch/SDSMT.LOCAL/bscott/benchmark_zarr_{uncompressed,blosc}.zarr` (8 timepoints, about 32 GB) from a raw OME-TIFF on `/mmfs2`, timing each write. |
| `extraction.py` | Times four ROI reads from a 4.3 GB raw OME-TIFF on `/mmfs2` through tifffile's zarr view (read only). |
| `logic.py` | Prints the fluorophores `parse_name` finds in one dataset folder name (pure regex, touches no files). |
| `extraction_plan_smoke.py` | Prints `get_extraction_plan` for one raw OME-TIFF on `/mmfs2`, under `__main__` only (read only). |

Other probes that were `test_*.py` outside `tests/`: `scripts/decon_probe.py` (prints "Starting..." and nothing else) and
`scratch/mp_probe.py` (two processes sharing a `multiprocessing.Value`, prints which worker got which counter).
