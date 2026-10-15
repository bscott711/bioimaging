"""Is a dataset's `final` output complete and readable?

Intermediates (and, in a duplicate pair, the other copy's outputs) are only
ever suggested for deletion when the surviving final output passes this.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

_FRAME = re.compile(r"_C(\d+)_T(\d+)\.tif$")


def _read_full(path: Path) -> None:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
    if arr.size == 0:
        raise ValueError("empty array")


CHECK_VERSION = 2  # bump when the checks change: cached results are keyed on it


def _verify_single_volumes(rows: list[tuple[str, int]], root: Path, read) -> tuple[bool, str]:
    """Bead/PSF scans and other one-volume outputs (`bead_GFP_488.ome.tif`):
    every non-MIP TIFF in the final dir is non-empty and decodes."""
    vols = [(r, s) for r, s in rows if "/" not in r and r.endswith((".tif", ".tiff")) and "_MIP_" not in r]
    if not vols:
        return False, "no *_C#_T#.tif frames or single-volume TIFF in final dir"
    if any(s == 0 for _r, s in vols):
        return False, "zero-length volume"
    try:
        for r, _s in vols:
            read(root / r)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable volume: {exc}"
    return True, f"{len(vols)} single volume(s) decoded"


def verify_tiff_series(
    rows: list[tuple[str, int]], root: Path, expected_timepoints: int | None, read=_read_full
) -> tuple[bool, str]:
    """`rows`: (relpath-below-root, size) of the final dir. Requires, per
    channel, contiguous timepoints 0..T-1 with the same T across channels (and
    == the registry's `actual_timepoints` when known), no empty files, and the
    first and last frame of every channel decodable."""
    frames: dict[int, dict[int, tuple[str, int]]] = defaultdict(dict)
    for rel, size in rows:
        name = rel.rsplit("/", 1)[-1]
        if name.endswith("_MIP_z.tif") or "/" in rel:
            continue
        m = _FRAME.search(name)
        if m:
            frames[int(m.group(1))][int(m.group(2))] = (rel, size)
    if not frames:
        return _verify_single_volumes(rows, root, read)
    counts = set()
    for c, by_t in sorted(frames.items()):
        ts = sorted(by_t)
        if ts != list(range(len(ts))):
            return False, f"channel {c}: timepoints not contiguous from 0 ({ts[0]}..{ts[-1]}, n={len(ts)})"
        if any(s == 0 for _r, s in by_t.values()):
            return False, f"channel {c}: zero-length frame"
        counts.add(len(ts))
    if len(counts) != 1:
        return False, f"channels disagree on timepoint count: {sorted(counts)}"
    (t,) = counts
    if expected_timepoints and t != expected_timepoints:
        return False, f"{t} timepoints on disk, registry expects {expected_timepoints}"
    try:
        for c, by_t in frames.items():
            for ti in (0, t - 1):
                read(root / by_t[ti][0])
    except Exception as exc:  # noqa: BLE001 - any decode failure = not verified
        return False, f"unreadable frame: {exc}"
    return True, f"{len(frames)} channel(s) x {t} timepoints, first/last frame decoded"


def _verify_live_progress(root: Path, expected_timepoints: int | None) -> tuple[bool, str] | None:
    """The live lane records completeness in `.opym_live.json`: state, n_t, n_c and
    every (t, c) written. Zarr writers skip all-zero chunks, so the chunk grid
    cannot prove completeness; this file can."""
    try:
        meta = json.loads((root / ".opym_live.json").read_text())
    except (OSError, ValueError):
        return None
    n_t, n_c = meta.get("n_t"), meta.get("n_c")
    if meta.get("state") != "complete":
        return False, f"live store state is {meta.get('state')!r}, not complete"
    want = {(t, c) for t in range(n_t) for c in range(n_c)}
    done = {tuple(x) for x in meta.get("done", [])}
    if want - done:
        return False, f"{len(want - done)} of {len(want)} (t, c) planes never written"
    if expected_timepoints and n_t != expected_timepoints:
        return False, f"{n_t} timepoints in store, registry expects {expected_timepoints}"
    return True, f"live store complete: {n_t} timepoints x {n_c} channels recorded written"


def verify_zarr(
    rows: list[tuple[str, int]], root: Path, expected_timepoints: int | None = None
) -> tuple[bool, str]:
    """Live-lane stores are checked against their progress file. Any other zarr must
    have every array's chunk-file count equal its chunk grid (zarr v2 .zarray or v3
    zarr.json) with no empty chunk -- conservative: a sparse store is held, never
    called complete. A file belongs to the deepest array directory containing it."""
    live = _verify_live_progress(root, expected_timepoints)
    if live is not None:
        return live
    meta_names = (".zarray", ".zattrs", ".zgroup", "zarr.json")
    arrays: dict[str, dict] = {}
    for rel, _s in rows:
        base = rel.rsplit("/", 1)[-1]
        if base not in (".zarray", "zarr.json"):
            continue
        try:
            meta = json.loads((root / rel).read_text())
        except (OSError, ValueError) as exc:
            return False, f"unreadable {rel}: {exc}"
        if base == "zarr.json" and meta.get("node_type") != "array":
            continue
        arrays[rel.rsplit("/", 1)[0] if "/" in rel else ""] = meta
    if not arrays:
        return False, "no zarr arrays found"
    chunks_found: dict[str, int] = dict.fromkeys(arrays, 0)
    ordered = sorted(arrays, key=len, reverse=True)
    for rel, size in rows:
        if rel.rsplit("/", 1)[-1] in meta_names:
            continue
        for adir in ordered:
            if adir == "" or rel.startswith(adir + "/"):
                if size > 0:
                    chunks_found[adir] += 1
                break
    for adir, meta in arrays.items():
        shape = meta["shape"]
        chunks = meta["chunks"] if "chunks" in meta else meta["chunk_grid"]["configuration"]["chunk_shape"]
        want = math.prod(math.ceil(s / c) for s, c in zip(shape, chunks, strict=True))
        if chunks_found[adir] != want:
            return False, f"array {adir or '.'}: {chunks_found[adir]} non-empty chunks, grid needs {want}"
    return True, f"{len(arrays)} array(s), chunk grids complete"


def verify_final(
    rows: list[tuple[str, int]], final_root: Path, final_is_zarr: bool,
    expected_timepoints: int | None, read=_read_full,
) -> tuple[bool, str]:
    if not rows:
        return False, "no final files on disk"
    if final_is_zarr:
        return verify_zarr(rows, final_root, expected_timepoints)
    return verify_tiff_series(rows, final_root, expected_timepoints, read)
