"""Loaders for everything the rules need. All read-only."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from backfill.storage_triage.rules import Label
from backfill.storage_triage.store import connect_ro


def overall_status(statuses: dict[str, str], signal_flag: str | None) -> str:
    """Mirror of opym-dashboard's `_overall_status` (a contract, not shared code)."""
    vals = list(statuses.values())
    if signal_flag == "blocked" and "failed" in vals:
        return "blocked"
    if "failed" in vals:
        return "failed"
    if statuses.get("mip_encode") == "done":
        return "done"
    if "running" in vals:
        return "running"
    return "in_progress" if vals else "pending"


def stage_facts(registry: sqlite3.Connection) -> tuple[dict[str, dict[str, str]], dict[str, str | None]]:
    stages: dict[str, dict[str, str]] = defaultdict(dict)
    reprocessed: dict[str, str | None] = {}
    for r in registry.execute("SELECT dataset_key, stage, status, finished_at FROM stage_status"):
        stages[r["dataset_key"]][r["stage"]] = r["status"]
        if r["stage"] in ("deskew", "mip_encode") and r["finished_at"]:
            cur = reprocessed.get(r["dataset_key"])
            reprocessed[r["dataset_key"]] = max(cur, r["finished_at"]) if cur else r["finished_at"]
    return stages, reprocessed


def load_labels(triage_db: Path) -> dict[str, Label]:
    """dataset_key -> Label. `source` is missing from databases written before
    the dashboard recorded it; those labels predate Accept-all ever having
    anything to accept (no auto_triage.sqlite3 existed), so they are human."""
    conn = connect_ro(triage_db)
    if conn is None:
        return {}
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(triage)")}
        src = "source" if "source" in cols else "'human'"
        return {
            r["dataset_key"]: Label(r["label"], r["source"] or "human", r["updated_at"])
            for r in conn.execute(f"SELECT dataset_key, label, {src} AS source, updated_at FROM triage")
        }
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def load_auto(auto_db: Path) -> dict[str, dict]:
    conn = connect_ro(auto_db)
    if conn is None:
        return {}
    try:
        return {r["dataset_key"]: dict(r) for r in conn.execute("SELECT * FROM auto_triage")}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def load_seglab_protected(seg_db: Path) -> dict[str, list[str]]:
    """dataset_key -> reasons. The 27 cells with a chosen training timepoint
    (`tstar`) and every cell that passed the training-set gate."""
    conn = connect_ro(seg_db)
    if conn is None:
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    try:
        for r in conn.execute("SELECT dataset_key FROM tstar"):
            out[r[0]].append("seglab training cell (tstar)")
        for r in conn.execute("SELECT dataset_key FROM cells WHERE gate='pass'"):
            if "seglab training cell (tstar)" not in out.get(r[0], []):
                out[r[0]].append("seglab gate=pass (training-set candidate)")
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return dict(out)


def load_psf_sources(psf_dir: Path) -> set[str]:
    """Scan names listed in `*.manifest.json` -> matched against dataset leaf names later."""
    names: set[str] = set()
    for m in sorted(psf_dir.glob("*.manifest.json")):
        try:
            names.update(json.loads(m.read_text()).get("scans", []))
        except (OSError, ValueError):
            continue
    return names


def load_pinned(path: Path) -> list[str]:
    """One path prefix per line (`#` comments): datasets the PI pins by hand."""
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def protected_reasons(
    key: str, leaf_dir: str, seglab: dict[str, list[str]], psf_scans: set[str], pinned: list[str]
) -> list[str]:
    reasons = list(seglab.get(key, []))
    leaf = leaf_dir.rstrip("/").rsplit("/", 1)[-1]
    if leaf in psf_scans and "psf" in leaf_dir.lower():
        reasons.append("source scan of an averaged PSF (manifest)")
    reasons += [f"pinned by PI ({p})" for p in pinned if leaf_dir.startswith(p.rstrip("/"))]
    return reasons
