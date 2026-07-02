#!/usr/bin/env python
"""Time (and optionally MATLAB-profile) run_gpu_pipeline.m directly, to measure
the effect of the Phase 1 fixes described in
.claude/plans/we-have-just-finished-reflective-flute.md.

Calls matlab.engine directly, the same way
opym_local/tests/test_gpu_pipeline_regression.py does -- bypassing the JSON
queue/watchdog, since that's pure orchestration overhead we're not measuring
here. Runs N times back-to-back on a fixed crop so PSF/OTF cache warmth
(Phase 1.1) shows up in run 2..N vs run 1.

Requirements (same as the regression test):
    module load matlab/R2024b
    <bioimaging venv>/bin/python bioimaging/scripts/profile_phase1.py

Usage:
    # synthetic fixture (fast, always available)
    python scripts/profile_phase1.py --runs 5

    # real single-timepoint crop (needs OPYM_REGRESSION_REAL_DATA_DIR set,
    # see generate_real_data_fixture.py)
    python scripts/profile_phase1.py --runs 5 --real-data

    # also dump MATLAB's built-in line-level profiler report per run
    python scripts/profile_phase1.py --runs 3 --debug

To compare before/after Phase 1, run this script once per checkout:

    # baseline (pre-Phase-1)
    (cd ../opym_local && git checkout c7dd337 -- src/opym/run_gpu_pipeline.m src/opym/run_petakit_server.m)
    git checkout 3e2c480 -- run_pipeline_cli.py   # (in bioimaging)
    python scripts/profile_phase1.py --runs 5 --real-data > /tmp/before.txt

    # Phase 1 applied (current main in both repos)
    (cd ../opym_local && git checkout main -- src/opym/run_gpu_pipeline.m src/opym/run_petakit_server.m)
    git checkout main -- run_pipeline_cli.py
    python scripts/profile_phase1.py --runs 5 --real-data > /tmp/after.txt

    diff /tmp/before.txt /tmp/after.txt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

try:
    import matlab.engine  # noqa: F401  (must precede numpy/zarr, see conftest note)
except ImportError as e:
    raise SystemExit(
        "matlab.engine not importable -- run this from bioimaging's venv "
        "with `module load matlab/R2024b` active."
    ) from e

import numpy as np
import zarr

OPYM_LOCAL_DIR = Path(__file__).resolve().parents[2] / "opym_local"
FIXTURES_DIR = OPYM_LOCAL_DIR / "tests" / "fixtures"
PSF_PATH = FIXTURES_DIR / "psf" / "averaged_psf.tif"
SYNTHETIC_RAW_PATH = FIXTURES_DIR / "synthetic_raw_volume.npy"
OPYM_SRC_DIR = OPYM_LOCAL_DIR / "src" / "opym"
PETAKIT_ROOT = Path(os.environ.get("PETAKIT_ROOT", "/cm/shared/apps_local/petakit5d"))

# Must match the corrected production defaults (Phase 1.4: DeconIter=25).
PIPELINE_KWARGS = dict(
    xyPixelSize=0.136,
    z_step_um=0.3,
    dzPSF=0.1,
    DeconIter=25,
    RLMethod="simple",
    SkewAngle=60.0,
    interpMethod="cubic",
)


def _start_engine() -> "matlab.engine.MatlabEngine":
    eng = matlab.engine.start_matlab("-nodisplay")
    eng.gpuDevice(1, nargout=1)
    eng.addpath(str(OPYM_SRC_DIR), nargout=0)
    eng.addpath(str(OPYM_SRC_DIR / "patches"), nargout=0)
    if eng.exist("XR_deskew_rotate_data_wrapper", "file", nargout=1) == 0:
        setup_m = PETAKIT_ROOT / "setup.m"
        if not setup_m.exists():
            raise SystemExit(f"PetaKit5D setup.m not found at {setup_m}")
        eng.run(str(setup_m), nargout=0)
    return eng


def _write_shm_zarr(path: Path, array: np.ndarray) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    zarr.save(str(path), array)


def _wait_for_zarr(path: Path, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    marker = path / ".zarray"
    while time.time() < deadline:
        if marker.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for pipeline output at {path}")


def _load_raw_volume(real_data: bool) -> np.ndarray:
    if not real_data:
        return np.load(SYNTHETIC_RAW_PATH)

    real_dir = os.environ.get("OPYM_REGRESSION_REAL_DATA_DIR")
    if not real_dir:
        raise SystemExit(
            "--real-data requires OPYM_REGRESSION_REAL_DATA_DIR to be set "
            "(see opym_local/tests/fixtures/generate_real_data_fixture.py)."
        )
    raw_path = Path(real_dir) / "raw_crop.npy"
    if not raw_path.exists():
        raise SystemExit(f"Real-data raw crop not found at {raw_path}")
    return np.load(raw_path)


def run_once(eng, run_dir: Path, raw_volume: np.ndarray, tag: str, debug: bool) -> float:
    shm_in = run_dir / "in.zarr"
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_fn = out_dir / f"{tag}.tif"

    _write_shm_zarr(shm_in, raw_volume)

    kwargs = dict(PIPELINE_KWARGS, debug=debug)
    name_value_args = []
    for k, v in kwargs.items():
        name_value_args.append(k)
        name_value_args.append(v)

    start = time.time()
    eng.run_gpu_pipeline(
        str(shm_in), str(output_fn), str(PSF_PATH), *name_value_args, nargout=1
    )
    _wait_for_zarr(out_dir / f"{tag}.zarr")
    elapsed = time.time() - start

    if debug:
        prof_dir = Path("/dev/shm/petakit_jobs/profiling") / f"{tag}_html"
        print(f"    MATLAB profiler report: {prof_dir}/index.html")

    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=5, help="number of back-to-back runs on the same crop")
    parser.add_argument("--real-data", action="store_true", help="use the real single-timepoint fixture instead of synthetic")
    parser.add_argument("--debug", action="store_true", help="also dump MATLAB's built-in line-level profiler report per run")
    parser.add_argument("--work-dir", type=Path, default=Path("/dev/shm/profile_phase1_scratch"))
    args = parser.parse_args()

    raw_volume = _load_raw_volume(args.real_data)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting MATLAB engine (crop shape {raw_volume.shape}, DeconIter={PIPELINE_KWARGS['DeconIter']})...")
    eng = _start_engine()

    times = []
    try:
        for i in range(args.runs):
            tag = f"run{i}"
            elapsed = run_once(eng, args.work_dir, raw_volume, tag, args.debug)
            times.append(elapsed)
            warm_note = " (cache should be warm from here on -- Phase 1.1)" if i == 1 else ""
            print(f"  run {i}: {elapsed:.3f}s{warm_note}")
    finally:
        eng.quit()

    print()
    print(f"n={len(times)}  min={min(times):.3f}s  max={max(times):.3f}s  mean={sum(times)/len(times):.3f}s")
    if len(times) > 1:
        print(f"first run: {times[0]:.3f}s  subsequent mean: {sum(times[1:])/len(times[1:]):.3f}s")


if __name__ == "__main__":
    main()
