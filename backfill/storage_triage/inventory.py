"""Classify and size every byte a registered dataset owns.

Classes (each file/dir gets exactly one):
  raw           the acquisition itself (MMStack TIFFs + metadata, or the raw zarr store)
  final         the registry's deskew `output_path` -- the Decon->DSR (or DSR) result
  provenance    small records the dashboard/viewers read: mip_movies, MIPs, json/txt
  intermediate  regenerable pipeline products (crops, DSR_nodecon next to a decon final, ...)
  viewer_export `viewer/*_dsr.ome.zarr` when it is not itself the final (a copy of the DSR)
  unknown       anything we do not recognise -- never suggested for deletion
"""

from __future__ import annotations

import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

RAW, FINAL, PROV, INTER, VIEWER, UNKNOWN = (
    "raw", "final", "provenance", "intermediate", "viewer_export", "unknown",
)
CLASSES = (RAW, FINAL, PROV, INTER, VIEWER, UNKNOWN)

DEFAULT_MIRROR_ROOT = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/outputs")

_RAW_FILE = re.compile(r"(\.ome\.tiff?|\.tiff?)$|_metadata\.txt$|^AcqSettings\.txt$", re.I)
_PROV_FILE = re.compile(r"\.(json|txt|mat|log|cxc|csv|md)$", re.I)
# Directories the pipeline writes; anything under them that is not final or
# provenance is a regenerable intermediate.
_DERIVED_TOPS = {"processed_ngff", "processed_tiff_series_split", "decon_stage", "zarr_mirror"}
_PROV_TOPS = {"mip_movies", "decon_qc"}
# Wrapper dirs whose loose files and child dirs are separate deletable units.
_WRAPPERS = {"processed_tiff_series_split", "decon_stage", "zarr_mirror", "Decon"}
MIN_SUGGEST_BYTES = 50 * 2**20  # units smaller than this are inventoried, never suggested


@dataclass
class Scan:
    files: list[tuple[str, str, int, int, str]] = field(default_factory=list)  # tree, rel, size, mtime_ns, cls
    units: dict[tuple[str, str, str], list[int]] = field(default_factory=dict)  # (tree, unit, cls) -> [bytes, n, max_mtime]
    exists: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _under(parts: tuple[str, ...], prefix: tuple[str, ...] | None) -> bool:
    return bool(prefix) and parts[: len(prefix)] == prefix


def classify(parts: tuple[str, ...], is_dir: bool, final_rel: tuple[str, ...] | None) -> str:
    """`parts`: path relative to the tree root (leaf dir or mirror dir)."""
    if _under(parts, final_rel):
        return FINAL
    name = parts[-1]
    top = parts[0]
    if len(parts) == 1 and not is_dir:
        if _RAW_FILE.search(name):
            return RAW
        return PROV if _PROV_FILE.search(name) else UNKNOWN
    if top in _PROV_TOPS:
        return PROV
    if top == "viewer":
        if len(parts) >= 2 and parts[1].endswith("_dsr.ome.zarr"):
            return VIEWER
        return PROV
    if top in _DERIVED_TOPS:
        if "MIPs" in parts:
            return PROV
        if len(parts) >= 2 and not is_dir and _PROV_FILE.search(name) and not name.endswith("_T000.tif"):
            return PROV
        return INTER
    return UNKNOWN


def unit_of(parts: tuple[str, ...], is_dir_entry_path: bool = False) -> str:
    """Deletable-unit label for a file at `parts`: its top dir, descending
    through wrapper dirs so loose files and child dirs become separate units."""
    cur: list[str] = []
    for i, p in enumerate(parts[:-1]):
        cur.append(p)
        if p not in _WRAPPERS:
            return "/".join(cur)
        # descended into a wrapper: next component is a child dir or a loose file
        if i + 1 == len(parts) - 1:  # next is the file itself
            return "/".join(cur) + "/*"
    return parts[0] if len(parts) == 1 else "/".join(parts[:-1]) + "/*"


def _walk(root: Path):
    """Yield (parts, is_dir, size, mtime_ns) for everything below root (files and
    dirs). Directories are yielded so empty ones classify; sizes are files only."""
    stack = [((), root)]
    while stack:
        parts, d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError as exc:
            yield parts, True, -1, str(exc)  # signal unreadable dir
            continue
        for e in entries:
            p = parts + (e.name,)
            try:
                if e.is_dir(follow_symlinks=False):
                    yield p, True, 0, 0
                    stack.append((p, Path(e.path)))
                else:
                    st = e.stat(follow_symlinks=False)
                    yield p, False, st.st_size, st.st_mtime_ns
            except OSError as exc:
                yield p, True, -1, str(exc)


def scan_tree(
    scan: Scan, tree: str, root: Path, final_rel: tuple[str, ...] | None, skip_dirs: set[str]
) -> None:
    scan.exists[tree] = root.is_dir()
    if not root.is_dir():
        return
    for parts, is_dir, size, mtime in _walk(root):
        if size == -1:
            scan.errors.append(f"{tree}:{'/'.join(parts)}: {mtime}")
            continue
        if parts and parts[0] in skip_dirs:
            continue
        if is_dir:
            continue
        cls = classify(parts, False, final_rel)
        rel = "/".join(parts)
        if cls in (RAW, FINAL):
            scan.files.append((tree, rel, size, mtime, cls))
        key = (tree, unit_of(parts) if cls != FINAL else "/".join(final_rel or parts[:1]), cls)
        agg = scan.units.setdefault(key, [0, 0, 0])
        agg[0] += size
        agg[1] += 1
        agg[2] = max(agg[2], mtime)


def _rel_parts(path: str | None, root: Path) -> tuple[str, ...] | None:
    if not path:
        return None
    try:
        return Path(path).relative_to(root).parts or None
    except ValueError:
        return None


def scan_dataset(
    row: dict, mirror_root: Path, other_leaf_dirs: set[str]
) -> Scan:
    """Scan one registry dataset: its leaf dir, its mirror output dir (when
    the leaf was read-only to the backfill), and -- for zarr acquisitions --
    the raw `.ome.zarr` store, which lives beside the leaf, not inside it."""
    scan = Scan()
    leaf = Path(row["leaf_dir"])
    mirror = mirror_root / str(leaf).lstrip("/")
    final = row.get("final_path")
    skip = set()
    # a leaf dir may nest another registered dataset; never count its bytes
    for other in other_leaf_dirs:
        try:
            skip.add(Path(other).relative_to(leaf).parts[0])
        except ValueError:
            pass
    for tree, root in (("leaf", leaf), ("mirror", mirror)):
        scan_tree(scan, tree, root, _rel_parts(final, root), skip)
    master = row.get("master_file") or ""
    if master.endswith(".zarr") or ".zarr/" in master:
        store = Path(master[: master.index(".zarr") + 5])
        scan.exists["master"] = store.is_dir()
        if store.is_dir():
            for parts, is_dir, size, mtime in _walk(store):
                if size == -1:
                    scan.errors.append(f"master:{'/'.join(parts)}: {mtime}")
                elif not is_dir:
                    scan.files.append(("master", store.name + "/" + "/".join(parts), size, mtime, RAW))
    return scan


def write_scan(conn: sqlite3.Connection, key: str, scan: Scan, final_rel: str | None) -> None:
    conn.execute("DELETE FROM files WHERE dataset_key=?", (key,))
    conn.execute("DELETE FROM units WHERE dataset_key=?", (key,))
    conn.executemany(
        "INSERT INTO files VALUES (?,?,?,?,?,?)", [(key, *f) for f in scan.files]
    )
    conn.executemany(
        "INSERT INTO units VALUES (?,?,?,?,?,?,?)",
        [(key, t, u, c, b, n, m) for (t, u, c), (b, n, m) in scan.units.items()],
    )
    conn.execute(
        "INSERT OR REPLACE INTO scans VALUES (?,?,?,?)",
        (key, datetime.now(UTC).isoformat(), final_rel, ",".join(sorted(t for t, ok in scan.exists.items() if ok))),
    )
    conn.commit()


def registry_rows(registry: sqlite3.Connection) -> list[dict]:
    """Datasets joined with their deskew output path -- the registry's own
    record of where the final output went, never re-derived from dir names."""
    cur = registry.execute(
        """
        SELECT d.dataset_key, d.root, d.leaf_dir, d.master_file, d.signal_flag,
               d.expected_timepoints, d.actual_timepoints, d.discovered_at,
               sk.output_path AS final_path
        FROM datasets d
        LEFT JOIN stage_status sk ON sk.dataset_key = d.dataset_key AND sk.stage = 'deskew'
        """
    )
    names = [c[0] for c in cur.description]
    return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


def scan_all(
    conn: sqlite3.Connection, rows: list[dict], mirror_root: Path, workers: int = 12,
    refresh: bool = False, log=print,
) -> None:
    done = set() if refresh else {r[0] for r in conn.execute("SELECT dataset_key FROM scans")}
    todo = [r for r in rows if r["dataset_key"] not in done]
    leafs = {r["leaf_dir"] for r in rows}
    log(f"[scan] {len(todo)} to scan ({len(done)} cached)")

    def work(r):
        others = {l for l in leafs if l != r["leaf_dir"] and l.startswith(r["leaf_dir"].rstrip("/") + "/")}
        return r, scan_dataset(r, mirror_root, others)

    with ThreadPoolExecutor(workers) as ex:
        for i, (r, scan) in enumerate(ex.map(work, todo), 1):
            write_scan(conn, r["dataset_key"], scan, r.get("final_path"))
            if i % 25 == 0 or i == len(todo):
                log(f"[scan] {i}/{len(todo)}")
