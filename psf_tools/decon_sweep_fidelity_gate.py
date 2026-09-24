#!/usr/bin/env python3
"""Fidelity gate (plan Step 4): confirms the sweep's out-of-band 'prod' arm
reproduces the already-approved production DSR output for Cell_005 T000
before any other variant is trusted. Compares shape, Pearson r, and max|delta|.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

PROD_DSR = Path(
    "/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/"
    "20260917-SVO-memNG-mScar2xFYVE-FLM-Macropinocytosis/Cell_005/"
    "decon_stage/Decon/DSR_decon"
)


def compare(sweep_prod_tif: Path, prod_tif: Path) -> dict:
    """Stays in uint16 / int32 throughout -- no whole-array float64 copies --
    and short-circuits on exact equality (both arms are deterministic given
    identical params, so bit-identical output is the expected case and is
    far cheaper to detect than a correlation coefficient). Falls back to a
    correlation computed via integer sums (no float64 temporaries) only if
    the arrays differ."""
    a = tifffile.imread(sweep_prod_tif)
    b = tifffile.imread(prod_tif)
    same_shape = a.shape == b.shape
    if not same_shape:
        return {"same_shape": False, "a_shape": a.shape, "b_shape": b.shape}

    if np.array_equal(a, b):
        return {
            "same_shape": True,
            "shape": a.shape,
            "exact_match": True,
            "pearson_r": 1.0,
            "max_abs_delta": 0.0,
            "mean_abs_delta": 0.0,
            "pass": True,
        }

    # Differ somewhere -- quantify with int32 (safe: uint16 range squared
    # fits in int64 accumulation) instead of promoting the whole volume to
    # float64.
    ai = a.astype(np.int32)
    bi = b.astype(np.int32)
    delta = ai - bi
    max_abs_delta = float(np.abs(delta).max())
    mean_abs_delta = float(np.abs(delta).mean())
    n = ai.size
    sa = ai.sum(dtype=np.int64)
    sb = bi.sum(dtype=np.int64)
    saa = np.sum(ai.astype(np.int64) * ai, dtype=np.int64)
    sbb = np.sum(bi.astype(np.int64) * bi, dtype=np.int64)
    sab = np.sum(ai.astype(np.int64) * bi, dtype=np.int64)
    cov = sab - sa * sb / n
    var_a = saa - sa * sa / n
    var_b = sbb - sb * sb / n
    r = float(cov / np.sqrt(var_a * var_b)) if var_a > 0 and var_b > 0 else float("nan")
    return {
        "same_shape": True,
        "shape": a.shape,
        "exact_match": False,
        "pearson_r": r,
        "max_abs_delta": max_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "pass": r > 0.999,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, required=True)
    args = ap.parse_args()

    for ch in ("C0", "C1"):
        sweep_tif = args.sweep_root / "stage" / "Decon_prod" / "DSR" / f"Cell_005_{ch}_T000.tif"
        prod_tif = PROD_DSR / f"Cell_005_{ch}_T000.tif"
        print(f"\n=== {ch} ===")
        print(f"sweep: {sweep_tif}")
        print(f"prod:  {prod_tif}")
        result = compare(sweep_tif, prod_tif)
        for k, v in result.items():
            print(f"  {k}: {v}")
        if not result.get("pass", False):
            print(f"  !!! FIDELITY GATE FAILED for {ch}")


if __name__ == "__main__":
    main()
