#!/usr/bin/env python3
"""Compare plain Richardson-Lucy vs OTF-masked Wiener (OMW) deconvolution on the
FIRST TIMEPOINT of a real cell dataset, through the (now OMW-capable) unified
GPU pipeline.

Motivation: run_gpu_pipeline.m historically ignored ``RLMethod`` and always ran
plain RL, so requesting ``omw`` in production silently gave plain RL. Now that
the pipeline generates a real OMW (Wiener-Butterworth) back projector and its
``wienerAlpha``/``OTFCumThresh``/``hannWinBounds`` knobs are exposed end to end,
this harness stages T=0 of one channel once and submits one pipeline job per
decon variant (plain RL, and OMW at one or more wienerAlpha values). Each job
runs Decon -> Deskew/Rotate -> Z-trim and writes a final deskewed volume; those
volumes are what the RL-vs-OMW comparison artifact is built from.

Only T=0 is processed (fast; the streaking is already visible in a single
volume). Everything mirrors the staging/submit/poll pattern already validated in
``psf_tools/sweep_deskew_angles.py``.

Run:  uv run python -m psf_tools.omw_rl_comparison --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

# run_pipeline_cli (ROI/z-step helpers) lives at the bioimaging repo root;
# opym is the editable dependency. Mirror sweep_deskew_angles.py's path setup so
# this runs both as a module and as a bare script.
_BIOIMAGING_ROOT = Path(__file__).resolve().parents[1]
if str(_BIOIMAGING_ROOT) not in sys.path:
    sys.path.append(str(_BIOIMAGING_ROOT))

from run_pipeline_cli import auto_detect_rois, get_z_step  # noqa: E402
from opym.petakit import submit_pipeline_job, wait_for_job  # noqa: E402
from opym.utils import orient_zyx_for_dsr  # noqa: E402

DEFAULT_TARGET = Path(
    "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/"
    "20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0.ome.tif"
)
# This bead PSF predates dz_psf metadata tagging, so its z-step is passed
# explicitly (bead stack acquired at 0.1 um/step). Same PSF used by the
# validated angle sweep.
DEFAULT_PSF = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif")
DEFAULT_PSF_DZ_UM = 0.1

SHM_DIR = Path("/dev/shm/opym_jobs")
QUEUE_DIR = Path("/dev/shm/petakit_jobs/queue")


def build_variants(alpha_sweep: bool) -> list[dict]:
    """The decon variants to compare. Plain RL is the baseline; OMW at the stock
    wienerAlpha (0.005) is the like-for-like 'what omw should have been doing'
    case; larger alpha = stronger Wiener regularization (less streak/noise
    amplification, softer). ``alpha_sweep`` widens the OMW alpha range so the
    artifact can show the sharpness/streak trade-off directly."""
    variants = [
        {"name": "RL_simple_25it", "rl_method": "simple", "iterations": 25},
        {"name": "OMW_a0.005", "rl_method": "omw", "iterations": 2, "wiener_alpha": 0.005},
        {"name": "OMW_a0.02", "rl_method": "omw", "iterations": 2, "wiener_alpha": 0.02},
    ]
    if alpha_sweep:
        variants += [
            {"name": "OMW_a0.001", "rl_method": "omw", "iterations": 2, "wiener_alpha": 0.001},
            {"name": "OMW_a0.05", "rl_method": "omw", "iterations": 2, "wiener_alpha": 0.05},
        ]
    return variants


def stage_first_timepoint(target: Path, channel: int, roi_side: str):
    """Read T=0 of ``channel`` from the raw OME-TIF, crop the auto-detected
    optical-FOV ROI, and orient it into the (ny, nx, nz) layout the GPU pipeline
    expects. Returns (oriented_crop, z_step_um, roi)."""
    store = tifffile.imread(str(target), aszarr=True)
    z = zarr.open(store, mode="r")
    if z.ndim != 5:
        raise ValueError(f"Expected 5D (T,Z,C,Y,X) data, got shape {z.shape}")

    z_step_um = get_z_step(target) or 0.3
    top_roi, bot_roi = auto_detect_rois(z)
    roi = top_roi if roi_side == "top" else bot_roi
    if not roi:
        roi = (slice(None), slice(None))

    # Channel axis is whichever of axes 1/2 is smaller (C=2 vs Z=175 here).
    c_axis = 1 if z.shape[1] < z.shape[2] else 2
    if c_axis == 2:
        cropped = np.array(z[0, :, channel, roi[0], roi[1]])
    else:
        cropped = np.array(z[0, channel, :, roi[0], roi[1]])

    oriented = orient_zyx_for_dsr(cropped)
    return oriented, z_step_um, roi


def run_comparison(
    target: Path,
    psf_path: Path,
    psf_dz_um: float,
    output_base: Path,
    channel: int,
    roi_side: str,
    sheet_angle_deg: float,
    variants: list[dict],
    poll_interval: int = 3,
) -> dict:
    """Stage T=0 once, then submit + wait for one pipeline job per variant.
    Returns a manifest dict (also written to ``output_base/manifest.json``)."""
    oriented, z_step_um, roi = stage_first_timepoint(target, channel, roi_side)
    print(
        f"Staged T=0, channel={channel}, roi_side={roi_side}, "
        f"oriented (ny,nx,nz)={oriented.shape}, z_step={z_step_um} um"
    )
    SHM_DIR.mkdir(parents=True, exist_ok=True)
    output_base.mkdir(parents=True, exist_ok=True)

    results = []
    for v in variants:
        name = v["name"]
        print(f"\n=== {name} ({v['rl_method']}, {v['iterations']} it"
              + (f", alpha={v['wiener_alpha']}" if "wiener_alpha" in v else "")
              + ") ===")

        # Re-stage per variant: run_gpu_pipeline_async rmdir's the shm input
        # after each job, so each variant needs its own fresh copy.
        shm_path = SHM_DIR / f"omwcmp_{name}.zarr"
        zarr.save(str(shm_path), oriented)

        output_file = output_base / name / "DSR" / f"omwcmp_{name}.zarr"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        job_path = submit_pipeline_job(
            output_file=output_file,
            shm_path=shm_path,
            psf_paths=[str(psf_path)],
            z_step_um=z_step_um,
            sheet_angle_deg=sheet_angle_deg,
            dz_psf=psf_dz_um,
            iterations=v["iterations"],
            rl_method=v["rl_method"],
            wiener_alpha=v.get("wiener_alpha"),
            save_zarr=True,
            queue_dir=QUEUE_DIR,
        )
        ok = wait_for_job(job_path, poll_interval=poll_interval)
        print(f"  -> {'OK' if ok else 'FAILED'}: {output_file}")
        results.append({**v, "output_file": str(output_file), "ok": bool(ok)})

    manifest = {
        "target": str(target),
        "psf_path": str(psf_path),
        "psf_dz_um": psf_dz_um,
        "channel": channel,
        "roi_side": roi_side,
        "roi": [[roi[0].start, roi[0].stop], [roi[1].start, roi[1].stop]]
        if roi[0].start is not None else None,
        "z_step_um": z_step_um,
        "sheet_angle_deg": sheet_angle_deg,
        "variants": results,
    }
    manifest_path = output_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_ok = sum(r["ok"] for r in results)
    print(f"\nDone: {n_ok}/{len(results)} variants succeeded. Manifest: {manifest_path}")
    return manifest


def load_dsr(output_file: Path) -> np.ndarray:
    """Load a pipeline DSR output (zarr) as a numpy array."""
    return np.array(zarr.open(str(output_file), mode="r"))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Raw master OME-TIF.")
    p.add_argument("--psf", type=Path, default=DEFAULT_PSF)
    p.add_argument("--psf-dz-um", type=float, default=DEFAULT_PSF_DZ_UM)
    p.add_argument("--channel", type=int, default=0, help="Camera/channel index to deconvolve.")
    p.add_argument("--roi-side", choices=["top", "bottom"], default="top")
    p.add_argument("--sheet-angle-deg", type=float, default=60.0,
                   help="Deskew sheet angle (90 - theta); the validated production value is 60.")
    p.add_argument("--alpha-sweep", action="store_true",
                   help="Add extra OMW wienerAlpha values (0.001, 0.05) to the comparison.")
    p.add_argument("--output-base", type=Path, default=None,
                   help="Where to write per-variant DSR outputs. Defaults to "
                        "<dataset>/cell_MMStack_Pos0/Decon/OMW_vs_RL_T0.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.target.exists():
        raise FileNotFoundError(f"Target not found: {args.target}")
    if not args.psf.exists():
        raise FileNotFoundError(f"PSF not found: {args.psf}")

    output_base = args.output_base or (
        args.target.parent / "cell_MMStack_Pos0" / "Decon" / "OMW_vs_RL_T0"
    )
    run_comparison(
        target=args.target,
        psf_path=args.psf,
        psf_dz_um=args.psf_dz_um,
        output_base=output_base,
        channel=args.channel,
        roi_side=args.roi_side,
        sheet_angle_deg=args.sheet_angle_deg,
        variants=build_variants(args.alpha_sweep),
    )


if __name__ == "__main__":
    main()
