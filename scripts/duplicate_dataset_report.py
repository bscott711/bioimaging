#!/usr/bin/env python3
"""Reports datasets discovered under more than one backfill root -- e.g. the
same acquisition physically copied to both /mmfs1 and /mmfs2's DataUpload
trees. Discovery deliberately does not dedupe across roots (dataset_key is
the absolute leaf path, so a copy on a different filesystem gets its own
row and its own full crop/deskew/MIP run) -- see
opym_local/src/opym/discovery.py's `discover_leaf_datasets` docstring.

Read-only, reporting only: makes no registry or filesystem changes. Run
after any backfill pass to refresh the numbers:

    uv run python scripts/duplicate_dataset_report.py [output.csv]
"""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_REGISTRY_PATH = "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3"
DEFAULT_OUTPUT = "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/duplicates.csv"


def overall_status(by_key_stages: dict, dataset_key: str, signal_flag: str | None) -> str:
    """Mirrors opym-dashboard/app/registry_reader.py's `_overall_status` --
    kept as a local copy rather than imported, matching that repo's own
    "schema contract, not shared code" convention for reading this registry.
    """
    statuses = list(by_key_stages.get(dataset_key, {}).values())
    if signal_flag == "blocked" and "failed" in statuses:
        return "blocked"
    if "failed" in statuses:
        return "failed"
    if by_key_stages.get(dataset_key, {}).get("mip_encode") == "done":
        return "done"
    if "running" in statuses:
        return "running"
    if statuses:
        return "in_progress"
    return "pending"


def relative_leaf(leaf_dir: str, roots: list[str]) -> str:
    """The path segment identifying a dataset independent of which root it
    was discovered under -- the longest (most specific) matching root wins,
    so a dataset registered under a narrower one-off `--roots` subdirectory
    (see the stale-root note below) still groups with its sibling under the
    full default root.
    """
    for root in roots:
        if leaf_dir == root or leaf_dir.startswith(root.rstrip("/") + "/"):
            rel = leaf_dir[len(root):].lstrip("/")
            return rel if rel else Path(leaf_dir).name
    return leaf_dir


def main() -> None:
    registry_path = os.environ.get("OPYM_BACKFILL_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    conn = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    datasets = list(conn.execute("SELECT * FROM datasets"))
    by_key_stages: dict[str, dict[str, str]] = defaultdict(dict)
    for s in conn.execute("SELECT * FROM stage_status"):
        by_key_stages[s["dataset_key"]][s["stage"]] = s["status"]

    # Sorted longest-first so a narrower stale root (see the note printed
    # below) doesn't shadow the real, full-corpus root it's nested under.
    roots = sorted({d["root"] for d in datasets}, key=len, reverse=True)

    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for d in datasets:
        groups[relative_leaf(d["leaf_dir"], roots)].append(d)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "relative_leaf_path", "root", "leaf_dir", "overall_status",
                "signal_flag", "expected_timepoints", "actual_timepoints",
                "discovered_at",
            ]
        )
        for key in sorted(dupes):
            for d in sorted(dupes[key], key=lambda x: x["root"]):
                w.writerow(
                    [
                        key, d["root"], d["leaf_dir"],
                        overall_status(by_key_stages, d["dataset_key"], d["signal_flag"]),
                        d["signal_flag"] or "", d["expected_timepoints"] or "",
                        d["actual_timepoints"] or "", d["discovered_at"],
                    ]
                )

    total_rows = sum(len(v) for v in dupes.values())
    redundant = total_rows - len(dupes)
    disagreeing = sum(
        1 for rows in dupes.values()
        if len({overall_status(by_key_stages, d["dataset_key"], d["signal_flag"]) for d in rows}) > 1
    )
    print(f"{len(dupes)} leaf paths discovered under more than one root")
    print(f"{total_rows} total rows involved, {redundant} redundant copies")
    print(f"{disagreeing} duplicate groups have copies at DIFFERENT overall_status (worth a look first)")
    print(f"wrote {out_path}")

    # Cosmetic but worth flagging: register_dataset()'s ON CONFLICT clause
    # never refreshes `root` after first insert (opym_local/src/opym/
    # registry.py), so a dataset first registered via a one-off narrower
    # `--roots <subdir>` run keeps that subdirectory as its `root` forever,
    # even once later full-corpus passes rediscover it under the real
    # default root too. Surfaced here since it makes by-root grouping (this
    # report's own `roots` list, and the dashboard's by_root stat) show
    # extra, misleadingly narrow roots.
    stale = sorted(
        r for r in roots
        if any(other != r and r.startswith(other.rstrip("/") + "/") for other in roots)
    )
    if stale:
        print(f"\nNote: {len(stale)} root value(s) look like stale one-off subdirectory roots:")
        for r in stale:
            print(f"   {r}")


if __name__ == "__main__":
    main()
