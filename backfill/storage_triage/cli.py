"""opym-storage-triage -- evidence-backed storage delete SUGGESTIONS. Never deletes.

  scan     inventory every registered dataset (read-only walk of the filesystems)
  hash     SHA-256 the raw files of candidate duplicate copies (resumable; nice + idle I/O, optional --max-mbps)
  report   check finals, apply the rules, write the report directory
  verify   re-derive a previous report's rows from current disk state before you delete
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from backfill.storage_triage import dupes
from backfill.storage_triage.analyze import Analysis, analyze, check_finals, scan_ages
from backfill.storage_triage.inventory import DEFAULT_MIRROR_ROOT, registry_rows, scan_all
from backfill.storage_triage.report import write_report, write_verdicts
from backfill.storage_triage.store import connect, connect_ro

BASE = Path(os.environ.get("OPYM_BACKFILL_DIR", "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill"))
SEG_DB = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_seg/seg.sqlite3")
PSF_DIR = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF")
# `report` refuses to build suggestions from an inventory older than this (ST-05).
MAX_SCAN_AGE_HOURS = 72.0
NICE_INCREMENT = 10


def throttle(a: argparse.Namespace) -> list[str]:
    """Run at low priority: CPU nice and, where `ionice` exists, the idle I/O class, so the
    full-tree walk and the SHA-256 pass do not starve the pipeline sharing the filesystems.
    Opt out with --no-throttle. Returns what was applied (for the log)."""
    if getattr(a, "no_throttle", False):
        return []
    applied = []
    try:
        os.nice(NICE_INCREMENT)
        applied.append(f"nice +{NICE_INCREMENT}")
    except OSError:
        pass
    io = shutil.which("ionice")
    if io:
        r = subprocess.run([io, "-c", "3", "-p", str(os.getpid())], capture_output=True, check=False)
        if r.returncode == 0:
            applied.append("ionice idle")
    return applied


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
    print("[scan] throttle:", ", ".join(throttle(a)) or "off")
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
    print("[hash] throttle:", ", ".join(throttle(a)) or "off")
    hasher = dupes.sha256_file
    if a.max_mbps:
        hasher = functools.partial(dupes.sha256_file, limiter=dupes.RateLimiter(a.max_mbps * 2**20))
        print(f"[hash] capped at {a.max_mbps:g} MiB/s across all workers")
    dupes.hash_missing(conn, todo, workers=a.workers, hasher=hasher)


def _unique_dir(base: Path, stem: str) -> Path:
    """A report directory that never overwrites an earlier one: `verify` and the PI work
    from those files. Named to the second; a clash gets -2, -3, ..."""
    out, n = base / stem, 1
    while out.exists():
        n += 1
        out = base / f"{stem}-{n}"
    return out


def cmd_report(a):
    p = _paths(a)
    conn = connect(p["db"])
    reg = _open_registry(p["registry"])
    rows_l = registry_rows(reg)
    print("[report] throttle:", ", ".join(throttle(a)) or "off")
    if a.rescan:
        scan_all(conn, rows_l, Path(a.mirror_root), workers=a.workers, refresh=True)
    ages, unscanned = scan_ages(conn, [r["dataset_key"] for r in rows_l])
    stale = {k: h for k, h in ages.items() if h > a.max_scan_age_hours}
    if ages:
        print(f"[report] inventory age: newest {min(ages.values()):.1f} h, oldest {max(ages.values()) / 24:.1f} days; "
              f"{len(stale)} dataset(s) older than {a.max_scan_age_hours:g} h, {len(unscanned)} never scanned")
    if stale and not a.allow_stale:
        sys.exit(
            f"refusing to build suggestions from a stale inventory ({len(stale)} dataset(s) scanned more than "
            f"{a.max_scan_age_hours:g} h ago; oldest {max(stale.values()) / 24:.1f} days). Run "
            "`scan --refresh` first (or `report --rescan`), or pass --allow-stale to label the output stale."
        )
    check_finals(conn, rows_l, Path(a.mirror_root), workers=a.workers)
    A = _run_analysis(a, p, conn, reg)
    out = _unique_dir(Path(a.work_dir), datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    extra = {"report_dir": str(out), "oldest_scan_age_hours": round(max(ages.values()), 1) if ages else None,
             "stale_datasets": len(stale), "allow_stale": bool(a.allow_stale)}
    summary = write_report(A, out, {r["dataset_key"]: r for r in rows_l}, extra)
    if not a.no_verdicts:
        write_verdicts(A, Path(a.verdict_db), stale=sorted(stale))
    print(json.dumps(summary, indent=2))


def cmd_verify(a):
    """Re-derive the suggestions from current disk + labels and compare to a
    previous suggestions.csv. Run this immediately before deleting anything.

    Every dataset the CSV names is rescanned: the rows' own datasets AND the kept copy of
    every A1 row (and the other copies of their duplicate clusters). A deleted or changed
    keeper must fail the row, not be read back from a stale inventory."""
    p = _paths(a)
    conn = connect(p["db"])
    reg = _open_registry(p["registry"])
    print("[verify] throttle:", ", ".join(throttle(a)) or "off")
    with open(a.suggestions, newline="") as fh:
        old = list(csv.DictReader(fh))
    registry = {r["dataset_key"]: r for r in registry_rows(reg)}
    mirror = Path(a.mirror_root)
    keys = {r["dataset_key"] for r in old}
    keepers = {r["kept_copy"] for r in old if r.get("kept_copy")}

    # 1. os.stat the keeper's raw files against the inventory the hashes were taken from,
    #    BEFORE the rescan overwrites that inventory with whatever is on disk now
    keeper_bad: dict[str, str] = {}
    raw_before = dupes.load_raw(conn)
    for kk in sorted(keepers):
        if kk not in registry:
            keeper_bad[kk] = "kept copy is no longer in the registry"
            continue
        r = registry[kk]
        files = [(f, f.abspath(r["leaf_dir"], mirror, r["master_file"])) for f in raw_before.get(kk, [])]
        if not files:
            keeper_bad[kk] = "kept copy has no raw files in the inventory"
            continue
        problems = dupes.keeper_file_problems(files)
        if problems:
            keeper_bad[kk] = problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")

    # 2. rescan the rows' datasets, the keepers and the rest of their duplicate clusters
    touch = set(keys) | keepers
    for comp in dupes.clusters(dupes.candidate_pairs(raw_before)):
        if comp & touch:
            touch |= comp
    rows = [r for k, r in registry.items() if k in touch]
    scan_all(conn, rows, mirror, workers=a.workers, refresh=True)
    check_finals(conn, rows, mirror, workers=a.workers)
    A = _run_analysis(a, p, conn, reg)
    now = {(s.tier, s.path): s for s in A.suggestions}
    bad = 0
    for r in old:
        s = now.get((r["tier"], r["path"]))
        kept = r.get("kept_copy") or ""
        if kept and kept in keeper_bad:
            print(f"KEEPER CHANGED       {r['tier']} {r['path']}  ({kept}: {keeper_bad[kept]})")
            bad += 1
        elif s is None:
            print(f"NO LONGER SUGGESTED  {r['tier']} {r['path']}")
            bad += 1
        elif s.fingerprint != r["fingerprint"]:
            print(f"CHANGED ON DISK      {r['tier']} {r['path']}  ({r['fingerprint']} -> {s.fingerprint})")
            bad += 1
        elif kept and s.kept_copy != kept:
            print(f"KEEPER CHANGED       {r['tier']} {r['path']}  (kept copy is now {s.kept_copy or 'none'}, was {kept})")
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
    ap.add_argument("--no-throttle", action="store_true",
                    help="run at normal CPU/I-O priority (default: nice +10 and idle I/O class)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--workers", type=int, default=6)
    s.add_argument("--refresh", action="store_true"); s.set_defaults(fn=cmd_scan)
    h = sub.add_parser("hash"); h.add_argument("--workers", type=int, default=4)
    h.add_argument("--max-mbps", type=float, default=None,
                   help="cap total hashing read rate, MiB/s across all workers (default: uncapped)")
    h.add_argument("--dry-run", action="store_true"); h.set_defaults(fn=cmd_hash)
    r = sub.add_parser("report"); r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-scan-age-hours", type=float, default=MAX_SCAN_AGE_HOURS,
                   help="refuse an inventory older than this (default %(default)g)")
    r.add_argument("--allow-stale", action="store_true",
                   help="build the report from an old inventory anyway; verdict rows are labelled stale")
    r.add_argument("--rescan", action="store_true", help="re-walk every dataset first (scan --refresh)")
    r.add_argument("--verdict-db", default=str(BASE / "storage_verdicts.sqlite3"),
                   help="where opym-dashboard reads verdicts from (default: next to registry.sqlite3)")
    r.add_argument("--no-verdicts", action="store_true"); r.set_defaults(fn=cmd_report)
    v = sub.add_parser("verify"); v.add_argument("suggestions"); v.add_argument("--workers", type=int, default=4)
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
