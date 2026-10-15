"""opym-storage-triage -- evidence-backed storage delete SUGGESTIONS. Never deletes.

  scan     inventory every registered dataset (read-only walk of the filesystems)
  hash     SHA-256 the raw files of candidate duplicate copies (resumable, throttled)
  report   check finals, apply the rules, write the report directory
  verify   re-derive a previous report's rows from current disk state before you delete
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from backfill.storage_triage import dupes
from backfill.storage_triage.analyze import Analysis, analyze, check_finals
from backfill.storage_triage.inventory import DEFAULT_MIRROR_ROOT, registry_rows, scan_all
from backfill.storage_triage.report import write_report, write_verdicts
from backfill.storage_triage.store import connect, connect_ro

BASE = Path(os.environ.get("OPYM_BACKFILL_DIR", "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill"))
SEG_DB = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_seg/seg.sqlite3")
PSF_DIR = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF")


def _paths(a: argparse.Namespace) -> dict:
    work = Path(a.work_dir)
    return dict(
        db=work / "storage_triage.sqlite3",
        registry=Path(a.registry),
        triage=Path(a.triage_db or Path(a.registry).parent / "triage.sqlite3"),
        auto=Path(a.auto_triage_db or work / "auto_triage_scratch.sqlite3"),
        pinned=work / "protected.txt",
    )


def _open_registry(p: Path):
    reg = connect_ro(p)
    if reg is None:
        sys.exit(f"registry not found: {p}")
    return reg


def _run_analysis(a, p, conn, reg) -> Analysis:
    return analyze(
        conn, reg, triage_db=p["triage"], auto_db=p["auto"], seg_db=Path(a.seg_db), psf_dir=Path(a.psf_dir),
        pinned_file=p["pinned"], mirror_root=Path(a.mirror_root),
    )


def cmd_scan(a):
    p = _paths(a)
    reg = _open_registry(p["registry"])
    conn = connect(p["db"])
    scan_all(conn, registry_rows(reg), Path(a.mirror_root), workers=a.workers, refresh=a.refresh)


def cmd_hash(a):
    p = _paths(a)
    conn = connect(p["db"])
    reg = _open_registry(p["registry"])
    rows = {r["dataset_key"]: r for r in registry_rows(reg)}
    raw = dupes.load_raw(conn)
    pairs = dupes.candidate_pairs(raw)
    todo = [(f, f.abspath(rows[f.dataset_key]["leaf_dir"], Path(a.mirror_root), rows[f.dataset_key]["master_file"]))
            for f in dupes.files_to_hash(raw, pairs)]
    print(f"[hash] {len(pairs)} candidate pairs, {len(dupes.clusters(pairs))} clusters, {len(todo)} raw files")
    if a.dry_run:
        print(f"[hash] dry run: {sum(f.size for f, _ in todo) / 2**40:.2f} TiB across candidate files")
        return
    dupes.hash_missing(conn, todo, workers=a.workers)


def cmd_report(a):
    p = _paths(a)
    conn = connect(p["db"])
    reg = _open_registry(p["registry"])
    rows_l = registry_rows(reg)
    check_finals(conn, rows_l, Path(a.mirror_root), workers=a.workers)
    A = _run_analysis(a, p, conn, reg)
    out = Path(a.work_dir) / datetime.now().strftime("%Y-%m-%d")
    summary = write_report(A, out, {r["dataset_key"]: r for r in rows_l}, {"report_dir": str(out)})
    if not a.no_verdicts:
        write_verdicts(A, Path(a.verdict_db))
    import json
    print(json.dumps(summary, indent=2))


def cmd_verify(a):
    """Re-derive the suggestions from current disk + labels and compare to a
    previous suggestions.csv. Run this immediately before deleting anything."""
    p = _paths(a)
    conn = connect(p["db"])
    reg = _open_registry(p["registry"])
    old = list(csv.DictReader(open(a.suggestions)))
    keys = sorted({r["dataset_key"] for r in old})
    rows = [r for r in registry_rows(reg) if r["dataset_key"] in set(keys)]
    # a changed raw file must be re-hashed before its duplicate claim stands
    scan_all(conn, rows, Path(a.mirror_root), workers=a.workers, refresh=True)
    A = _run_analysis(a, p, conn, reg)
    check_finals(conn, rows, Path(a.mirror_root), workers=a.workers)
    A = _run_analysis(a, p, conn, reg)
    now = {(s.tier, s.path): s for s in A.suggestions}
    bad = 0
    for r in old:
        s = now.get((r["tier"], r["path"]))
        if s is None:
            print(f"NO LONGER SUGGESTED  {r['tier']} {r['path']}")
            bad += 1
        elif s.fingerprint != r["fingerprint"]:
            print(f"CHANGED ON DISK      {r['tier']} {r['path']}  ({r['fingerprint']} -> {s.fingerprint})")
            bad += 1
    print(f"{len(old) - bad}/{len(old)} rows still hold" + ("" if not bad else f"; {bad} must NOT be deleted"))
    sys.exit(1 if bad else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="opym-storage-triage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(BASE / "registry.sqlite3"))
    ap.add_argument("--triage-db", default=None)
    ap.add_argument("--auto-triage-db", default=None)
    ap.add_argument("--seg-db", default=str(SEG_DB))
    ap.add_argument("--psf-dir", default=str(PSF_DIR))
    ap.add_argument("--mirror-root", default=str(DEFAULT_MIRROR_ROOT))
    ap.add_argument("--work-dir", default=str(BASE / "storage_triage"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--workers", type=int, default=12)
    s.add_argument("--refresh", action="store_true"); s.set_defaults(fn=cmd_scan)
    h = sub.add_parser("hash"); h.add_argument("--workers", type=int, default=8)
    h.add_argument("--dry-run", action="store_true"); h.set_defaults(fn=cmd_hash)
    r = sub.add_parser("report"); r.add_argument("--workers", type=int, default=6)
    r.add_argument("--verdict-db", default=str(BASE / "storage_verdicts.sqlite3"),
                   help="where opym-dashboard reads verdicts from (default: next to registry.sqlite3)")
    r.add_argument("--no-verdicts", action="store_true"); r.set_defaults(fn=cmd_report)
    v = sub.add_parser("verify"); v.add_argument("suggestions"); v.add_argument("--workers", type=int, default=6)
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
