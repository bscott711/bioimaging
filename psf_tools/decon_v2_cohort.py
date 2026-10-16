#!/usr/bin/env python3
"""Run the decon-v2 trial's `prod` control and the combined arm (N2V + haze floor + large synthetic PSF) on the segmentation cohort,
membrane channel only, at the timepoint the masks were drawn on (+ optional neighbours for a stability check).

  seglab dv2-cells                                                  # (CORE) writes $OPYM_SEG_ROOT/dv2_cohort/cells.json
  DECON_V2_WORKSPACE=<ws> uv run python -m psf_tools.decon_v2_cohort run --cells 0902_cell_004,... [--ts-around 2] [--gpu 1]

Per cell, each step skipped when its output exists (rerun to resume): stage -> N2V (trained on that cell's own frames, up to 5
timepoints spread through the series, T* among them) -> haze floor of the denoised volume -> `prod` and `n2v_synL_bg` through the
out-of-band MATLAB driver -> gate (`prod` or `prod_cubic` must reproduce the STORED volume, r >= 0.9999; the one that does sets the combined arm's interpolation) -> shape check -> manifest.
A cell that fails the gate or the shape check is reported and its volumes must not be used: the reviewed masks live on the stored
geometry. Nothing touches `DSR_decon`, the production queue or the services. GPU work waits while an acquisition holds the live lease.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from psf_tools import decon_v2_trial as T

CAREAMICS_PY = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/decon_sweeps/envs/careamics/bin/python")
CONTROLS = ["prod", "prod_cubic"]   # which of them reproduces the stored volume says how it was resampled
ARMS = ["n2v_synL_bg"]
LEASE = Path(os.environ.get("PETAKIT_JOBS_DIR", "/dev/shm/petakit_jobs")) / "LIVE_LEASE.json"
GATE_R = 0.9999


def wait_for_live_lane(poll_s: float = 15.0) -> None:
    announced = False
    while LEASE.exists() and time.time() - LEASE.stat().st_mtime <= 60.0:
        if not announced:
            print("live lease active: pausing GPU work until the acquisition ends", flush=True)
            announced = True
        time.sleep(poll_s)


def train_timepoints(t_star: int, available: list[int], n: int = 5) -> list[int]:
    """Up to ``n`` timepoints spread through the series, ``t_star`` among them (N2V learns the noise of this cell's own frames)."""
    if len(available) <= n:
        return sorted(available)
    pick = {available[int(round(i))] for i in np.linspace(0, len(available) - 1, n)}
    pick.add(t_star)
    while len(pick) > n:                       # keep t_star, drop the spread point nearest to it
        pick.discard(min(pick - {t_star}, key=lambda x: abs(x - t_star)))
    return sorted(pick)


def stage_n2v(cell: str, table: dict, ts: list[int], gpu: str, log: Path) -> None:
    from backfill.pipeline import build_decon_staging_dir

    sp = T.spec(cell)
    out_dir = T.stage_dir(cell, "stage_n2v")
    names = [f"{cell}_C0_T{t:03d}.tif" for t in ts]
    if all((out_dir / n).is_file() for n in names):
        return
    train_ts = train_timepoints(sp.t, table["timepoints"])
    train_dir = T.stage_dir(cell, "n2v_train")
    build_decon_staging_dir(sp.stores, train_dir, dataset_prefix=cell, timepoints=train_ts)
    train = [train_dir / f"{cell}_C0_T{t:03d}.tif" for t in train_ts]
    cmd = [str(CAREAMICS_PY), "-m", "psf_tools.n2v_denoise", "--train", *map(str, train), "--targets",
           *[str(T.stage_dir(cell) / n) for n in names], "--outs", *[str(out_dir / n) for n in names],
           "--work", str(T.cell_dir(cell) / "n2v_work_C0"), "--gpu", "0"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    with log.open("a") as fh:
        rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[1]))
    if rc:
        raise RuntimeError(f"{cell}: N2V failed (exit {rc}); see {log}")


def run_cell(cid: str, table: dict, ts_around: int, gpu: str) -> dict:
    t0 = time.time()
    c = table[cid]
    sp = T.spec(cid)
    ts = sorted({sp.t, *[t for t in range(sp.t - ts_around, sp.t + ts_around + 1) if t in c["timepoints"]]})
    T.SPECS[cid] = T.CellSpec(sp.t, sp.cohort_id, sp.stores, sp.z_step_um, sp.reference, sp.channels, tuple(ts))
    T.cell_dir(cid).mkdir(parents=True, exist_ok=True)
    wait_for_live_lane()
    if not any(T.stage_dir(cid).glob("*.tif")):
        T.stage_cell(cid)
    stage_n2v(cid, c, ts, gpu, T.cell_dir(cid) / "n2v_C0.log")
    if not any(T.stage_dir(cid, "stage_n2v_bg").glob("*.tif")):
        T.stage_n2v_bg_cell(cid)
    wait_for_live_lane()

    def arm(name):
        return T.arm_table(T.load_psfs(), cid)[name]

    def missing(names):
        return [a for a in names if not all(T.dsr_tif(cid, arm(a), ch, t).is_file() for ch in sp.channels for t in ts)]

    def sweep(names):
        todo = missing(names)
        if todo:
            rc = T.run_matlab(T.make_variants(cid, todo), T.cell_dir(cid) / "sweep_matlab.log", gpu)
            if rc:
                raise RuntimeError(f"{cid}: MATLAB sweep exit {rc}")

    sweep(CONTROLS)                                  # the two controls first: they decide how the combined arm is resampled
    T.make_variants(cid, CONTROLS)
    gates = {a: T.gate(cid, a) for a in CONTROLS}
    ok_arms = [a for a in CONTROLS if all(v["pearson_r"] >= GATE_R for v in gates[a].values())]
    interp = {"prod": None, "prod_cubic": "cubic"}[ok_arms[0]] if ok_arms else None
    T.SPECS[cid] = T.CellSpec(sp.t, sp.cohort_id, sp.stores, sp.z_step_um, sp.reference, sp.channels, sp.extra_ts, interp)
    wait_for_live_lane()
    sweep(ARMS)
    T.make_variants(cid, CONTROLS + ARMS)            # the variants file lists every arm (manifest / shape check read it)
    ok_gate = bool(ok_arms)
    shapes = T.check_shapes(cid)
    ok_shape = all(v == "ok" for v in shapes.values())
    T.write_manifest(cid)
    g = gates[ok_arms[0] if ok_arms else "prod"]
    res = dict(cid=cid, t=sp.t, ts=ts, z_step_um=sp.z_step_um, gate=g, gates={a: {c: v["pearson_r"] for c, v in gates[a].items()} for a in CONTROLS},
               gate_ok=ok_gate, matched_control=ok_arms[0] if ok_arms else None, interp=interp or "linear (current default)",
               shapes_ok=bool(ok_shape), combined=str(T.dsr_tif(cid, arm("n2v_synL_bg"), 0)), seconds=round(time.time() - t0))
    (T.cell_dir(cid) / "cohort_result.json").write_text(json.dumps(res, indent=1))
    print(f"{cid}: gate {'ok (' + res['matched_control'] + ', ' + res['interp'] + ')' if ok_gate else 'FAILED'} (r {[round(v['pearson_r'], 6) for v in g.values()]}), shapes {'ok' if ok_shape else 'BAD'}, "
          f"{res['seconds']} s", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--cohort", type=Path, default=Path(os.environ.get("OPYM_SEG_ROOT", "")) / "dv2_cohort" / "cells.json")
    ap.add_argument("--cells", required=True, help="comma-separated cohort ids")
    ap.add_argument("--ts-around", type=int, default=0, help="also process this many timepoints either side of T*")
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args(argv)
    table = json.loads(a.cohort.read_text())
    T.load_cohort(a.cohort)
    results = []
    for cid in a.cells.split(","):
        try:
            results.append(run_cell(cid, table, a.ts_around, a.gpu))
        except Exception as e:  # noqa: BLE001 - one bad cell must not stop the rest
            print(f"{cid}: ERROR {type(e).__name__}: {e}", flush=True)
            results.append(dict(cid=cid, error=f"{type(e).__name__}: {e}"))
    bad = [r["cid"] for r in results if "error" in r or not (r.get("gate_ok") and r.get("shapes_ok"))]
    print(f"done: {len(results) - len(bad)} ok, {len(bad)} not ok {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
