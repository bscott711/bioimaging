#!/usr/bin/env python3
"""
Wrap a bare Zarr v2 array (as produced by the *_processed.zarr pipeline output) in a
minimal OME-NGFF 0.4 `multiscales` group so ChimeraX's OME-Zarr opener will accept it.

Non-destructive: the wrapper is a new directory of *symlinks* to the original chunk
files plus three small JSON metadata files. The original store is never touched.

Source array axes (from .zarray shape + acquisition settings): (T, Z, C, Y, X) with C == 1.
The wrapper drops the singleton channel and presents a 4D (T, Z, Y, X) array as scale "0".
"""
import json
import os
import sys
from pathlib import Path

# ---- pixel / time calibration -------------------------------------------------
# PLACEHOLDERS.  stepSizeUm=0.1 comes from AcqSettings.txt (galvo slice step);
# lateral sampling was not recorded by the pipeline.  Fix in ChimeraX afterward
# with:  volume #N voxelSize <x>,<y>,<z>
T_SCALE = 7.0    # seconds between timepoints (AcqSettings timepointInterval)
Z_SCALE = 0.1    # micrometer  (AcqSettings stepSizeUm)
Y_SCALE = 0.1    # micrometer  (PLACEHOLDER)
X_SCALE = 0.1    # micrometer  (PLACEHOLDER)


def main(src: str, dst: str | None = None) -> None:
    src = Path(src).resolve()
    zarray_path = src / ".zarray"
    if not zarray_path.is_file():
        sys.exit(f"error: {src} has no .zarray - not a Zarr v2 array")

    meta = json.loads(zarray_path.read_text())
    shape = meta["shape"]
    chunks = meta["chunks"]
    if len(shape) != 5 or shape[2] != 1:
        sys.exit(f"error: expected 5D (T,Z,C,Y,X) with C==1, got shape {shape}")
    T, Z, C, Y, X = shape
    ct, cz, cc, cy, cx = chunks

    if dst is None:
        dst = src.with_name(src.name.replace(".zarr", "") + "_ome.zarr")
    dst = Path(dst).resolve()
    if dst.exists():
        sys.exit(f"error: {dst} already exists - remove it first")

    arr_dir = dst / "0"
    arr_dir.mkdir(parents=True)

    # --- group + multiscales metadata ---
    (dst / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    multiscales = {
        "multiscales": [
            {
                "version": "0.4",
                "name": src.name,
                "axes": [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale",
                             "scale": [T_SCALE, Z_SCALE, Y_SCALE, X_SCALE]},
                        ],
                    }
                ],
            }
        ]
    }
    (dst / ".zattrs").write_text(json.dumps(multiscales, indent=2))

    # --- 4D array metadata for scale "0" (T, Z, Y, X) ---
    # ChimeraX opens the store via zarr's FSStore(key_separator="/"), which rewrites
    # every chunk key to a nested "/"-separated path regardless of what .zarray says.
    # So the wrapper must use the nested chunk layout: 0/<t>/<z>/<y>/<x>.
    arr_meta = dict(meta)
    arr_meta["shape"] = [T, Z, Y, X]
    arr_meta["chunks"] = [ct, cz, cy, cx]
    arr_meta["dimension_separator"] = "/"
    (arr_dir / ".zarray").write_text(json.dumps(arr_meta, indent=2))

    # --- symlink every chunk:  0/<i>/<j>/0/0  ->  <src>/<i>.<j>.0.0.0 ---
    # original chunk grid is (nT, nZ, 1, 1, 1); wrapper grid is (nT, nZ, 1, 1)
    n_ct = -(-T // ct)   # ceil
    n_cz = -(-Z // cz)
    n = 0
    missing = 0
    for i in range(n_ct):
        for j in range(n_cz):
            old = src / f"{i}.{j}.0.0.0"
            if not old.exists():
                missing += 1
                continue
            leaf = arr_dir / str(i) / str(j) / "0"
            leaf.mkdir(parents=True, exist_ok=True)
            link = leaf / "0"
            os.symlink(os.path.relpath(old, leaf), link)
            n += 1
    print(f"wrapper written: {dst}")
    print(f"  chunk symlinks: {n}   (missing source chunks: {missing})")
    print(f"  open in ChimeraX:  open \"{dst}\"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: make_ngff_wrapper.py <src_processed.zarr> [<dst_ome.zarr>]")
    main(*sys.argv[1:3])
