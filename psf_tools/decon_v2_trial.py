#!/usr/bin/env python3
"""Decon v2 trial front end: stage one timepoint of a 0928 cell, write the
variants file the out-of-band sweep driver reads, launch it, and gate it.

Everything runs OUTSIDE the /dev/shm/petakit_jobs queue (the production
servers are untouched), through `opym/decon_sweep_driver.m`, on one GPU.
Nothing is written inside a dataset leaf and no `DSR_decon` is touched.

  uv run python -m psf_tools.decon_v2_trial stage    --cell Cell_004
  uv run python -m psf_tools.decon_v2_trial variants --cell Cell_004 --arms nodecon prod pad
  uv run python -m psf_tools.decon_v2_trial run      --cell Cell_004 --arms nodecon prod pad
  uv run python -m psf_tools.decon_v2_trial gate     --cell Cell_004 --variant prod

Handback constraint (CORE/seglab/HANDOFF_decon.md): every arm keeps production's
DSR geometry, or the reviewed masks drawn on it are void. `gate` checks the
`prod` arm against the existing viewer zarr; `check-shapes` checks all arms.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import zarr

from opym.decon_config import (
    DECON_DAMP_FACTOR,
    DECON_EDGE_EROSION,
    DECON_HANN_WIN_BOUNDS,
    DECON_OTF_CUM_THRESH,
    DECON_WIENER_ALPHA,
    DSR_INTERP_METHOD,
    XY_PIXEL_SIZE_UM,
)

WORKSPACE = Path(os.environ.get("DECON_V2_WORKSPACE", "/mmfs2/scratch/SDSMT.LOCAL/bscott/decon_sweeps/20261015_decon_v2"))
DATASET = Path(
    "/mmfs1/scratch/jacks.local/microscopy/20260928-SVO-memNG-mScar2xFYVE-FLM-Macropinocytosis"
)
BEAD_PSF = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260910_averaged_psf.tif")
# cell -> the timepoint the segmentation masks were drawn on (mip/results.json "t")
TRIAL_CELLS = {"Cell_004": 23, "Cell_005": 0, "Cell_001_002": 0}
COHORT_ID = {"Cell_004": "0928_C004", "Cell_005": "0928_C005", "Cell_001_002": "0928_C001_002"}
CHANNEL_PATTERNS = ["_C0_T", "_C1_T"]
CHANNELS = (0, 1)  # C0 = GFP 488 membrane, C1 = mScarlet 561 puncta
Z_STEP_UM = 0.5


@dataclass(frozen=True)
class CellSpec:
    """Everything cell-specific the trial needs. The three 0928 trial cells are built in; ``load_cohort`` adds the segmentation
    cohort (a table written by ``seglab dv2-cells``), keyed by cohort id, which is also the workspace directory name."""
    t: int                              # the timepoint the segmentation masks were drawn on
    cohort_id: str
    stores: tuple                       # raw OME-zarr per channel, C0 first
    z_step_um: float                    # scan step of the raw stack
    reference: dict | None = None       # the stored deskewed volume of t (None = the 0928 viewer zarr): what the prod arm must reproduce
    channels: tuple = CHANNELS
    extra_ts: tuple = ()                # more timepoints staged beside t (stability)
    interp: str | None = None           # DSR interpolation that reproduces the stored volume ('cubic' for datasets processed before
                                        # 2026-09-24); None = the pipeline's current default (linear), no override


SPECS: dict[str, CellSpec] = {
    c: CellSpec(t, COHORT_ID[c], tuple(DATASET / f"{c}_{lab}.ome.zarr" for lab in ("GFP_488", "mScarlet_561")), Z_STEP_UM)
    for c, t in TRIAL_CELLS.items()
}


def spec(cell: str) -> CellSpec:
    if cell not in SPECS:
        raise SystemExit(f"unknown cell {cell!r}; have {', '.join(sorted(SPECS))} (--cohort FILE adds the segmentation cohort)")
    return SPECS[cell]


def load_cohort(path: str | Path, channels: tuple = (0,)) -> list[str]:
    """Add the cells of a ``seglab dv2-cells`` table to ``SPECS`` (membrane channel only by default: the other channel is neither
    segmented nor needed, and N2V on it is the most expensive step). Returns the cohort ids."""
    from opym.metadata import parse_zarr_z_step_from_store

    table = json.loads(Path(path).read_text())
    labs = ("GFP_488", "mScarlet_561")
    for cid, c in table.items():
        if c["membrane_ch"] != 0:
            raise SystemExit(f"{cid}: membrane channel {c['membrane_ch']}; the cohort runner handles C0 only")
        stores = tuple(Path(c["raw"][labs[i]]) for i in channels)
        z = c["z_step_um"] or parse_zarr_z_step_from_store(stores[0])
        if not z:
            raise SystemExit(f"{cid}: no z step in the table and none in {stores[0]}")
        SPECS[cid] = CellSpec(int(c["t"]), cid, stores, float(z), c["reference"], tuple(channels))
        TRIAL_CELLS[cid] = int(c["t"])
        COHORT_ID[cid] = cid
    return list(table)
DZ_PSF_UM = 0.1
SKEW_DEG = 60.0
PAD_Z = 24  # >= half the decimated PSF support (17 planes bead; 33 planes large synthetic)
OPYM_DIR = Path(__file__).resolve().parents[2] / "opym_local" / "src" / "opym"

PROD = dict(
    decon=True, rl_method="omw", iters=2, alpha=DECON_WIENER_ALPHA, otf_thresh=DECON_OTF_CUM_THRESH,
    hann=list(DECON_HANN_WIN_BOUNDS), background=None, damp=DECON_DAMP_FACTOR,
)


def _omw(name: str, **kw) -> dict:
    return {**PROD, "name": name, **kw}


def arm_table(psfs: dict[str, str], cell: str | None = None) -> dict[str, dict]:
    """name -> variant. `psfs` maps a PSF id to a file (the per-channel
    {488, 561} pairs are written by synthetic_opm_psf.py into psfs.json)."""
    arms: dict[str, dict] = {
        "nodecon": {"name": "nodecon", "decon": False},
        "prod": _omw("prod"),
        "pad": _omw("pad", pad_z=PAD_Z, edge_erosion=0),
    }
    for a_tag, alpha in (("a05", 0.05), ("a02", 0.02)):
        for it in (2, 5, 10):
            n = f"{a_tag}_i{it}"
            arms[n] = _omw(n, alpha=alpha, iters=it, pad_z=PAD_Z, edge_erosion=0)
    # PSF-dependent arms: only present once the synthetic PSFs exist.
    for tag, key in (("syn", "syn"), ("synL", "synL"), ("synL_sa0", "synL_sa0")):
        if f"{key}_488" in psfs and f"{key}_561" in psfs:
            arms[tag] = _omw(tag, pad_z=PAD_Z, edge_erosion=0, psf_paths=[psfs[f"{key}_488"], psfs[f"{key}_561"]])
    if "synL_488" in psfs:
        # stage 2: the sharper settings with the large PSF (the bead-PSF `a*` arms found them noisier, not clearer)
        for n, alpha, it in (("synL_a10_i2", 0.10, 2), ("synL_a05_i2", 0.05, 2), ("synL_a05_i5", 0.05, 5),
                           ("synL_a05_i10", 0.05, 10), ("synL_a02_i5", 0.02, 5), ("synL_a02_i10", 0.02, 10)):
            arms[n] = _omw(n, alpha=alpha, iters=it, pad_z=PAD_Z, edge_erosion=0, psf_paths=[psfs["synL_488"], psfs["synL_561"]])
    if "synL_488" in psfs:
        # plain Richardson-Lucy (PetaKit 'simplified'): positivity-constrained iterations can restore some missing-cone
        # information that OMW's Wiener back projector cannot -- the direct test of "does decon move the haze"
        for n, it in (("synL_rl15", 15), ("synL_rl40", 40)):
            arms[n] = {"name": n, "decon": True, "rl_method": "simplified", "iters": it, "background": None, "pad_z": PAD_Z,
                       "edge_erosion": 0, "psf_paths": [psfs["synL_488"], psfs["synL_561"]]}
    if cell is not None and "synL_488" in psfs:
        # residual-haze background term: same synthetic large PSF, input with the smooth lower envelope removed
        arms["synL_bg"] = _omw("synL_bg", pad_z=PAD_Z, edge_erosion=0, stage_dir=str(stage_dir(cell, "stage_bg")),
                               psf_paths=[psfs["synL_488"], psfs["synL_561"]])
    # control for datasets processed before the 2026-09-24 switch to linear interpolation: production's settings with cubic
    arms["prod_cubic"] = _omw("prod_cubic", interp_method="cubic")
    if cell is not None:
        # self-supervised denoise (psf_tools.n2v_denoise) ahead of decon; `n2v_prod` changes ONLY the input
        arms["n2v_prod"] = _omw("n2v_prod", stage_dir=str(stage_dir(cell, "stage_n2v")))
        # views, not decon arms: the denoised input and the haze floor each through the plain DSR, to look at what they do alone
        arms["n2v_only"] = {"name": "n2v_only", "decon": False, "stage_dir": str(stage_dir(cell, "stage_n2v"))}
        arms["floor_only"] = {"name": "floor_only", "decon": False, "stage_dir": str(stage_dir(cell, "stage_floor"))}
        if "synL_488" in psfs:
            arms["n2v_synL"] = _omw("n2v_synL", pad_z=PAD_Z, edge_erosion=0, stage_dir=str(stage_dir(cell, "stage_n2v")),
                                    psf_paths=[psfs["synL_488"], psfs["synL_561"]])
            # all three together: denoise, then subtract the haze floor of the denoised volume, then decon with the large PSF
            arms["n2v_synL_bg"] = _omw("n2v_synL_bg", pad_z=PAD_Z, edge_erosion=0, stage_dir=str(stage_dir(cell, "stage_n2v_bg")),
                                       psf_paths=[psfs["synL_488"], psfs["synL_561"]])
    if cell in SPECS and SPECS[cell].interp:
        # every arm but the `prod` control resamples like the stored volume did (the control stays at the current default, so the
        # gate can tell which interpolation matches)
        for n, v in arms.items():
            if n not in ("prod", "prod_cubic"):
                v["interp_method"] = SPECS[cell].interp
    return arms


def cell_dir(cell: str) -> Path:
    return WORKSPACE / cell


def stage_dir(cell: str, which: str = "stage") -> Path:
    return cell_dir(cell) / which


def load_psfs() -> dict[str, str]:
    f = WORKSPACE / "psfs.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def stage_cell(cell: str) -> Path:
    from backfill.pipeline import build_decon_staging_dir

    sp = spec(cell)
    return build_decon_staging_dir(sp.stores, stage_dir(cell), dataset_prefix=cell, timepoints=sorted({sp.t, *sp.extra_ts}))


def stage_bg_cell(cell: str, open_um: float = 3.0, smooth_um: float = 3.0) -> Path:
    """`stage_bg/`: the staged TIFFs with the haze floor (psf_tools.decon_haze_bg) subtracted, same names."""
    from psf_tools.decon_haze_bg import subtract_haze

    out = stage_dir(cell, "stage_bg")
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(stage_dir(cell).glob("*.tif")):
        vol = tifffile.imread(f)
        tifffile.imwrite(out / f.name, subtract_haze(vol, xy_um=XY_PIXEL_SIZE_UM, dz_um=spec(cell).z_step_um, open_um=open_um,
                                                     smooth_um=smooth_um), compression="zlib")
        print(f"[stage_bg] {f.name}: median {np.median(vol):.0f} -> {np.median(tifffile.imread(out / f.name)):.0f}")
    return out


def stage_n2v_bg_cell(cell: str, open_um: float = 3.0, smooth_um: float = 3.0) -> Path:
    """`stage_n2v_bg/`: the DENOISED staged TIFFs (`stage_n2v/`) with the haze floor of the denoised volume subtracted.
    The floor is estimated after denoising because its lower envelope is then not pulled down by noise."""
    from psf_tools.decon_haze_bg import subtract_haze

    out = stage_dir(cell, "stage_n2v_bg")
    out.mkdir(parents=True, exist_ok=True)
    src = sorted(stage_dir(cell, "stage_n2v").glob("*.tif"))
    if not src:
        raise SystemExit(f"{cell}: no denoised frames in stage_n2v/; run psf_tools.n2v_denoise first")
    for f in src:
        tifffile.imwrite(out / f.name, subtract_haze(tifffile.imread(f), xy_um=XY_PIXEL_SIZE_UM, dz_um=spec(cell).z_step_um,
                                                     open_um=open_um, smooth_um=smooth_um), compression="zlib")
    return out


def stage_floor_cell(cell: str, open_um: float = 3.0, smooth_um: float = 3.0) -> Path:
    """`stage_floor/`: the haze floor itself (what `stage_bg` subtracts), as uint16 TIFFs with the staged names, so it can be
    deskewed and rotated like data and looked at."""
    from psf_tools.decon_haze_bg import haze_floor

    out = stage_dir(cell, "stage_floor")
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(stage_dir(cell).glob("*.tif")):
        floor = haze_floor(tifffile.imread(f), xy_um=XY_PIXEL_SIZE_UM, dz_um=spec(cell).z_step_um, open_um=open_um, smooth_um=smooth_um)
        tifffile.imwrite(out / f.name, np.clip(np.rint(floor), 0, np.iinfo(np.uint16).max).astype(np.uint16), compression="zlib")
    return out


def make_variants(cell: str, arm_names: list[str], extra: dict[str, dict] | None = None) -> Path:
    arms = {**arm_table(load_psfs(), cell), **(extra or {})}
    missing = [a for a in arm_names if a not in arms]
    if missing:
        raise SystemExit(f"unknown/unavailable arms {missing}; have {sorted(arms)}")
    variants = []
    for a in arm_names:
        v = dict(arms[a])
        if v.get("stage_dir") is None:
            v.pop("stage_dir", None)
        if v.get("psf_paths"):
            v["psf_paths"] = v["psf_paths"][:len(spec(cell).channels)]   # one PSF per channel pattern (the driver checks)
        variants.append(v)
    cfg = {
        "stage_dir": str(stage_dir(cell)),
        "psf_path": str(BEAD_PSF),
        "channel_patterns": CHANNEL_PATTERNS[:len(spec(cell).channels)],
        "xy_pixel_size": XY_PIXEL_SIZE_UM,
        "z_step_um": spec(cell).z_step_um,
        "dz_psf": DZ_PSF_UM,
        "skew_angle_deg": SKEW_DEG,
        "edge_erosion": DECON_EDGE_EROSION,
        "interp_method": DSR_INTERP_METHOD,
        "variants": variants,
    }
    out = cell_dir(cell) / "variants.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


def run_matlab(variants_json: Path, log: Path, gpu: str = "1") -> int:
    """One MATLAB session for the whole variant list (startup ~50 s). Same
    launch recipe as opym.local_gpu_worker, minus the queue."""
    cmd = (
        'if ! type module >/dev/null 2>&1; then . /etc/profile.d/modules.sh 2>/dev/null; fi; '
        f"module load matlab/R2024b && matlab -nodisplay -sd {OPYM_DIR} "
        f"-batch 'decon_sweep_driver(\"{variants_json}\")'"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PETAKIT_GPU_ID": gpu, "PETAKIT_CPUS": "8",
           "PETAKIT_ROOT": "/cm/shared/apps_local/petakit5d"}
    with log.open("a") as fh:
        return subprocess.call(["bash", "-lc", cmd], env=env, stdout=fh, stderr=subprocess.STDOUT)


def dsr_tif(cell: str, variant: dict, channel: int, t: int | None = None) -> Path:
    """Where the driver puts a variant's lab-frame volume (mirrors its layout); ``t`` defaults to the segmentation timepoint."""
    t = spec(cell).t if t is None else t
    name = variant["name"]
    fn = f"{cell}_C{channel}_T{t:03d}.tif"
    stg = Path(variant.get("stage_dir") or stage_dir(cell))
    if not variant.get("decon"):
        return stg / "DSR" / fn
    if variant.get("pad_z"):
        return stg.parent / f"pad{variant['pad_z']}_{name}" / f"Decon_{name}" / "DSR" / fn
    return stg / f"Decon_{name}" / "DSR" / fn


def read_variants(cell: str) -> dict[str, dict]:
    cfg = json.loads((cell_dir(cell) / "variants.json").read_text())
    return {v["name"]: v for v in cfg["variants"]}


def reference_volume(cell: str, channel: int) -> np.ndarray:
    """Production's volume for this cell/timepoint/channel: level 0 of the
    viewer zarr the segmentation was drawn on, as (z, y, x)."""
    sp = spec(cell)
    ref = sp.reference
    if ref is None:
        z = zarr.open(str(DATASET / cell / "viewer" / f"{cell}_dsr.ome.zarr"), mode="r")
        return np.asarray(z["0/0"][sp.t, channel])
    if channel != ref.get("c", 0) and ref["kind"] == "tiff":
        raise SystemExit(f"{cell}: the reference table holds channel 0 only")
    if ref["kind"] == "zarr":
        return np.asarray(zarr.open(ref["path"], mode="r")["0/0"][ref["t"], ref.get("c", channel)])
    return tifffile.imread(ref["path"])


def as_zyx(tif: np.ndarray, like_shape: tuple[int, ...]) -> np.ndarray:
    """PetaKit writes DSR as (y, x, z); the viewer zarr is (z, y, x)."""
    if tif.shape == like_shape:
        return tif
    cand = np.transpose(tif, (2, 0, 1))
    if cand.shape != like_shape:
        raise ValueError(f"DSR shape {tif.shape} cannot be matched to the reference {like_shape}")
    return cand


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    exact = bool(np.array_equal(a, b))
    r = 1.0 if exact else float(np.corrcoef(a, b)[0, 1])
    return {"exact": exact, "pearson_r": r, "max_abs_diff": float(np.abs(a - b).max()),
            "median_abs_diff": float(np.median(np.abs(a - b)))}


def gate(cell: str, variant_name: str = "prod") -> dict:
    v = read_variants(cell)[variant_name]
    out = {}
    for c in spec(cell).channels:
        ref = reference_volume(cell, c)
        got = as_zyx(tifffile.imread(dsr_tif(cell, v, c)), ref.shape)
        out[f"C{c}"] = {"shape": list(got.shape), "ref_shape": list(ref.shape), **compare(got, ref)}
        print(f"[gate] {cell} {variant_name} C{c}: {out[f'C{c}']}", flush=True)
    (cell_dir(cell) / f"gate_{variant_name}.json").write_text(json.dumps(out, indent=1))
    return out


def check_shapes(cell: str) -> dict:
    """Every arm's DSR volume must have the production geometry."""
    res = {}
    mf = cell_dir(cell) / "manifest.json"
    # every arm ever run on this cell (variants.json holds only the latest run's list)
    arms = {n: v["spec"] for n, v in json.loads(mf.read_text())["variants"].items()} if mf.is_file() else read_variants(cell)
    for name, v in arms.items():
        for c in spec(cell).channels:
            ref_shape = reference_volume(cell, c).shape
            p = dsr_tif(cell, v, c)
            if not p.is_file():
                res[f"{name}/C{c}"] = "MISSING"
                continue
            shape = tifffile.TiffFile(p).series[0].shape
            ok = tuple(shape) in (ref_shape, (ref_shape[1], ref_shape[2], ref_shape[0]))
            res[f"{name}/C{c}"] = "ok" if ok else f"SHAPE {shape} != {ref_shape}"
    bad = {k: s for k, s in res.items() if s != "ok"}
    print(json.dumps(res, indent=1))
    if bad:
        print(f"[check-shapes] PROBLEMS: {bad}", file=sys.stderr)
    return res


def write_manifest(cell: str) -> Path:
    """What the seglab scorer needs: per variant, the DSR TIFF of each channel (and the arm's settings)."""
    path = cell_dir(cell) / "manifest.json"
    # MERGE with what is there: each run rewrites variants.json with only the arms it ran, and earlier arms must stay
    out = json.loads(path.read_text()) if path.is_file() else {"variants": {}}
    out.update({"cell": cell, "cohort_id": spec(cell).cohort_id, "t": spec(cell).t})
    for name, v in read_variants(cell).items():
        files = {f"C{c}": str(dsr_tif(cell, v, c)) for c in spec(cell).channels}
        missing = [f for f in files.values() if not Path(f).is_file()]
        if missing:
            print(f"[manifest] {cell}/{name}: not on disk yet, left out ({missing[0]})")
            continue
        out["variants"][name] = {"files": files, "spec": v}
    p = cell_dir(cell) / "manifest.json"
    p.write_text(json.dumps(out, indent=1))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stage", "stage-bg", "stage-floor", "stage-n2v-bg", "variants", "run", "gate", "check-shapes", "manifest"])
    ap.add_argument("--cell", required=True, help="a 0928 trial cell, or a cohort id with --cohort")
    ap.add_argument("--cohort", type=Path, default=None, help="a `seglab dv2-cells` table: adds the segmentation cohort (C0 only)")
    ap.add_argument("--ts", default="", help="stage these timepoints beside the segmentation one (comma-separated)")
    ap.add_argument("--arms", nargs="*", default=["nodecon", "prod", "pad"])
    ap.add_argument("--variant", default="prod")
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args(argv)
    if a.cohort:
        load_cohort(a.cohort)
    sp0 = spec(a.cell)
    if a.ts:
        SPECS[a.cell] = CellSpec(sp0.t, sp0.cohort_id, sp0.stores, sp0.z_step_um, sp0.reference, sp0.channels,
                                 tuple(int(i) for i in a.ts.split(",")))
    cell_dir(a.cell).mkdir(parents=True, exist_ok=True)
    if a.cmd == "stage":
        print(stage_cell(a.cell))
    elif a.cmd == "stage-n2v-bg":
        print(stage_n2v_bg_cell(a.cell))
    elif a.cmd == "stage-floor":
        print(stage_floor_cell(a.cell))
    elif a.cmd == "stage-bg":
        print(stage_bg_cell(a.cell))
    elif a.cmd == "variants":
        print(make_variants(a.cell, a.arms))
    elif a.cmd == "run":
        vj = make_variants(a.cell, a.arms)
        rc = run_matlab(vj, cell_dir(a.cell) / "sweep_matlab.log", a.gpu)
        print(f"matlab exit {rc}")
        return rc
    elif a.cmd == "manifest":
        print(write_manifest(a.cell))
    elif a.cmd == "gate":
        gate(a.cell, a.variant)
    else:
        check_shapes(a.cell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
