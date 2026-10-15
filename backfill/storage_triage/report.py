"""Write the report directory: CSVs, explicit path lists, summary."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from backfill.storage_triage.analyze import Analysis, filesystem

TIERS = {
    "A1": "verified duplicate copies (byte-identical to a kept copy)",
    "A2": "regenerable pipeline intermediates (final output verified complete)",
    "A3": "viewer-export zarr copies of the DSR (napari/ChimeraX/3-D viewer read these; they fall back to the TIFFs)",
    "B": "datasets you labelled delete, every cross-check passed",
}


def write_report(A: Analysis, out_dir: Path, rows: dict[str, dict], extra: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "suggestions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "rule", "filesystem", "bytes", "nfiles", "class", "scope", "path",
                    "dataset_key", "kept_copy", "evidence", "fingerprint"])
        for s in sorted(A.suggestions, key=lambda s: (s.tier, -s.nbytes)):
            w.writerow([s.tier, s.rule, filesystem(s.path), s.nbytes, s.nfiles, s.cls, s.scope, s.path,
                        s.dataset_key, s.kept_copy, s.evidence, s.fingerprint])
    for tier in TIERS:
        with open(out_dir / f"paths_tier{tier}.txt", "w") as f:
            seen: set[str] = set()
            for s in A.suggestions:
                if s.tier != tier:
                    continue
                for p in (s.paths or [s.path]):
                    if p not in seen:
                        seen.add(p)
                        f.write(p + "\n")
    with open(out_dir / "review_queue.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "bytes", "filesystem", "dataset_key", "reasons"])
        for r in sorted(A.review, key=lambda r: -r.nbytes):
            w.writerow([r.rule, r.nbytes, filesystem(r.dataset_key), r.dataset_key, " | ".join(r.reasons)])
    with open(out_dir / "notes.txt", "w") as f:
        f.write("\n".join(A.notes) + "\n")

    by_tier: dict[str, dict] = {}
    for tier in TIERS:
        ss = [s for s in A.suggestions if s.tier == tier]
        fs: dict[str, int] = defaultdict(int)
        for s in ss:
            fs[filesystem(s.path)] += s.nbytes
        by_tier[tier] = {"what": TIERS[tier], "rows": len(ss), "datasets": len({s.dataset_key for s in ss}),
                         "bytes": sum(s.nbytes for s in ss), "by_filesystem": dict(fs)}
    summary = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "datasets_assessed": len(A.verdicts),
        "verdicts": {rule: sum(1 for v in A.verdicts.values() if v.rule == rule) for rule in ("P2", "P1", "D1", "R1", "K0")},
        "tiers": by_tier,
        "review_queue": {"datasets": len(A.review), "bytes": sum(r.nbytes for r in A.review)},
        "dup_clusters": len(A.cluster_info),
        "files_awaiting_hash": A.pending_files,
        "kept_bytes_registered": sum(A.kept_bytes.values()),
        **extra,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_verdicts(A: Analysis, db_path: Path) -> None:
    """`storage_verdicts.sqlite3`: one row per assessed dataset, read by
    opym-dashboard (a contract, not shared code). Rebuilt on every report."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    by_ds: dict[str, int] = defaultdict(int)
    dup: dict[str, str] = {}
    for s in A.suggestions:
        by_ds[s.dataset_key] += s.nbytes
        if s.rule == "DUP1":
            dup[s.dataset_key] = f"byte-identical to {s.kept_copy}"
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    conn.execute(
        "CREATE TABLE verdict (dataset_key TEXT PRIMARY KEY, rule TEXT NOT NULL, action TEXT NOT NULL, "
        "reasons TEXT NOT NULL, suggested_bytes INTEGER NOT NULL, generated_at TEXT NOT NULL)"
    )
    for k, v in A.verdicts.items():
        rule, reasons = (("DUP1", [dup[k]]) if k in dup else (v.rule, v.reasons))
        conn.execute("INSERT INTO verdict VALUES (?,?,?,?,?,?)",
                     (k, rule, v.action, "; ".join(reasons), by_ds.get(k, 0), now))
    conn.commit()
    conn.close()
    tmp.replace(db_path)   # atomic: the dashboard never sees a half-written file
