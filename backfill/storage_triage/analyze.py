"""Turn inventory + hashes + evidence into suggestions."""

from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from backfill.storage_triage import dupes, evidence
from backfill.storage_triage.completeness import verify_final
from backfill.storage_triage.inventory import (
    FINAL, INTER, MIN_SUGGEST_BYTES, PROV, RAW, UNKNOWN, VIEWER, classify, registry_rows,
)
from backfill.storage_triage.rules import (
    KEEP, REVIEW, SUGGEST_DELETE, Facts, Label, Verdict, decide,
)

ROOT_RANK = {"jacks.local/microscopy": 0, "mmfs1/scratch/SDSMT.LOCAL": 1, "mmfs2": 2}


@dataclass
class Suggestion:
    tier: str               # A1 dup copy | A2 intermediate | A3 viewer export | B human delete
    rule: str
    dataset_key: str
    path: str
    scope: str              # dir | files
    cls: str
    nbytes: int
    nfiles: int
    evidence: str
    kept_copy: str = ""
    paths: list[str] = field(default_factory=list)   # explicit expansion for the paths_*.txt files
    fingerprint: str = ""
    class_bytes: dict = field(default_factory=dict)  # class -> bytes, for whole-directory rows


@dataclass
class ReviewItem:
    dataset_key: str
    nbytes: int
    reasons: list[str]
    rule: str = "R1"


@dataclass
class Analysis:
    suggestions: list[Suggestion] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    kept_bytes: dict[str, int] = field(default_factory=dict)       # dataset_key -> bytes kept
    cluster_info: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pending_files: int = 0


def filesystem(path: str) -> str:
    return "mmfs2" if path.startswith("/mmfs2") else "mmfs1"


def _root_rank(leaf: str) -> int:
    for k, v in ROOT_RANK.items():
        if k in leaf:
            return v
    return 9


def _abs(tree: str, leaf: str, mirror_root: Path, rel: str, master: str | None) -> str:
    if tree == "leaf":
        return str(Path(leaf) / rel)
    if tree == "mirror":
        return str(mirror_root / leaf.lstrip("/") / rel)
    return str(Path(master[: master.index(".zarr") + 5]).parent / rel)


def _tree_root(tree: str, leaf: str, mirror_root: Path, master: str | None) -> Path:
    if tree == "leaf":
        return Path(leaf)
    if tree == "mirror":
        return mirror_root / leaf.lstrip("/")
    return Path(master[: master.index(".zarr") + 5])


def check_finals(
    conn: sqlite3.Connection, rows: list[dict], mirror_root: Path, workers: int = 6, log=print,
    only: set[str] | None = None,
) -> None:
    """Fill `final_checks` for datasets whose final output changed since the last check."""
    todo = []
    for r in rows:
        k = r["dataset_key"]
        if only is not None and k not in only:
            continue
        agg = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size),0), COALESCE(MAX(mtime_ns),0) FROM files WHERE dataset_key=? AND cls='final'",
            (k,),
        ).fetchone()
        fp = f"{agg[0]}:{agg[1]}:{agg[2]}"
        old = conn.execute("SELECT fingerprint FROM final_checks WHERE dataset_key=?", (k,)).fetchone()
        if not old or old[0] != fp:
            todo.append((r, fp))
    log(f"[verify-final] {len(todo)} dataset(s) to check")

    # sqlite connections are not thread-safe: gather file rows first, check in threads
    jobs = []
    for r, fp in todo:
        k, final = r["dataset_key"], r.get("final_path")
        if not final:
            conn.execute(
                "INSERT OR REPLACE INTO final_checks VALUES (?,?,?,?,?)",
                (k, fp, 0, "no deskew output recorded", datetime.now(UTC).isoformat()),
            )
            continue
        frows = conn.execute(
            "SELECT tree, rel, size FROM files WHERE dataset_key=? AND cls='final'", (k,)
        ).fetchall()
        picked = None
        for tree in ("leaf", "mirror"):
            root = _tree_root(tree, r["leaf_dir"], mirror_root, None)
            try:
                sub = Path(final).relative_to(root).as_posix()
            except ValueError:
                continue
            mine = [(f["rel"][len(sub) + 1:] if sub != "." else f["rel"], f["size"]) for f in frows
                    if f["tree"] == tree and (f["rel"] == sub or f["rel"].startswith(sub + "/"))]
            if mine:
                picked = (Path(final), mine)
                break
        jobs.append((r, fp, picked))

    def run(job):
        r, fp, picked = job
        if not picked:
            return r["dataset_key"], fp, False, "final output not found on disk"
        root, mine = picked
        ok, detail = verify_final(
            mine, root, root.name.endswith(".zarr"), r.get("actual_timepoints") or r.get("expected_timepoints")
        )
        return r["dataset_key"], fp, ok, detail

    with ThreadPoolExecutor(workers) as ex:
        for i, (k, fp, ok, detail) in enumerate(ex.map(run, jobs), 1):
            conn.execute(
                "INSERT OR REPLACE INTO final_checks VALUES (?,?,?,?,?)",
                (k, fp, int(ok), detail, datetime.now(UTC).isoformat()),
            )
            conn.commit()   # short write transactions: the hasher shares this database
            if i % 25 == 0:
                log(f"[verify-final] {i}/{len(jobs)}")
    conn.commit()


def _by_class(us: list[dict]) -> dict:
    out: dict[str, int] = defaultdict(int)
    for u in us:
        out[u["cls"]] += u["bytes"]
    return dict(out)


def _expand_units(units: list[dict], root: Path) -> list[str]:
    out: list[str] = []
    for u in units:
        p = root / u["unit"]
        if u["unit"].endswith("/*"):
            d = root / u["unit"][:-2]
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_file(follow_symlinks=False) and classify(
                            tuple(Path(u["unit"][:-2]).parts) + (e.name,), False, None
                        ) == u["cls"]:
                            out.append(e.path)
            except OSError:
                pass
        else:
            out.append(str(p))
    return out


def analyze(
    conn: sqlite3.Connection, registry: sqlite3.Connection, *, triage_db: Path, auto_db: Path,
    seg_db: Path, psf_dir: Path, pinned_file: Path, mirror_root: Path, now: datetime | None = None,
) -> Analysis:
    A = Analysis()
    rows = {r["dataset_key"]: r for r in registry_rows(registry)}
    stages, reprocessed = evidence.stage_facts(registry)
    labels = evidence.load_labels(triage_db)
    auto = evidence.load_auto(auto_db)
    seglab = evidence.load_seglab_protected(seg_db)
    psf_scans = evidence.load_psf_sources(psf_dir)
    pinned = evidence.load_pinned(pinned_file)
    final_ok = {r["dataset_key"]: (bool(r["ok"]), r["detail"]) for r in conn.execute("SELECT * FROM final_checks")}
    scanned = {r[0] for r in conn.execute("SELECT dataset_key FROM scans")}

    units: dict[str, list[dict]] = defaultdict(list)
    for u in conn.execute("SELECT * FROM units"):
        units[u["dataset_key"]].append(dict(u))
    total_bytes = {k: sum(u["bytes"] for u in us) for k, us in units.items()}
    # the "more complete copy" tie-break counts the acquisition, not derived outputs
    raw_bytes = {k: sum(u["bytes"] for u in us if u["cls"] == RAW) for k, us in units.items()}

    # ---- duplicate clusters (content-verified) -------------------------------------------
    raw = dupes.load_raw(conn)
    pairs = dupes.candidate_pairs(raw)
    hashes = dupes.load_hashes(conn)

    def digest_of(f: dupes.RawFile) -> str | None:
        r = rows[f.dataset_key]
        p = f.abspath(r["leaf_dir"], mirror_root, r["master_file"])
        return hashes.get((str(p), f.size, f.mtime_ns))

    comp_of: dict[str, int] = {}
    comps = dupes.clusters(pairs)
    for i, c in enumerate(comps):
        for k in c:
            comp_of[k] = i

    def facts_for(k: str) -> Facts:
        r = rows[k]
        sib = {s: labels[s] for s in comps[comp_of[k]] if s != k and s in labels} if k in comp_of else {}
        return Facts(
            key=k, leaf_dir=r["leaf_dir"],
            status=evidence.overall_status(stages.get(k, {}), r["signal_flag"]),
            signal_flag=r["signal_flag"], discovered_at=r["discovered_at"],
            reprocessed_at=reprocessed.get(k), label=labels.get(k), sibling_labels=sib,
            protected=evidence.protected_reasons(k, r["leaf_dir"], seglab, psf_scans, pinned),
            auto=auto.get(k), now=now,
        )

    keys = sorted(k for k in rows if k in scanned)
    verdict = {k: decide(facts_for(k)) for k in keys}
    A.verdicts = verdict
    nested = {
        k for k in keys
        if any(o != k and rows[o]["leaf_dir"].startswith(rows[k]["leaf_dir"].rstrip("/") + "/") for o in keys)
    }

    # ---- helpers to emit whole-copy / per-unit suggestions --------------------------------
    def whole_copy(k: str, tier: str, rule: str, why: str, kept: str = "") -> int:
        r = rows[k]
        us = units.get(k, [])
        has_unknown = any(u["cls"] == UNKNOWN for u in us)
        if k in nested or has_unknown:
            A.notes.append(
                f"{k}: {'nested dataset' if k in nested else 'unrecognised content'} -> suggested per known unit, not as a whole directory"
            )
        n = 0
        for tree in ("leaf", "mirror", "master"):
            tus = [u for u in us if u["tree"] == tree]
            if tree == "master":
                mf = [f for f in raw.get(k, []) if f.tree == "master"]
                if mf:
                    root = _tree_root("master", r["leaf_dir"], mirror_root, r["master_file"])
                    s = Suggestion(tier, rule, k, str(root), "dir", RAW, sum(f.size for f in mf), len(mf), why, kept,
                                   [str(root)], f"{len(mf)}:{sum(f.size for f in mf)}",
                                   {RAW: sum(f.size for f in mf)})
                    A.suggestions.append(s); n += s.nbytes
                continue
            if not tus:
                continue
            root = _tree_root(tree, r["leaf_dir"], mirror_root, None)
            if k not in nested and not has_unknown:
                nb, nf = sum(u["bytes"] for u in tus), sum(u["nfiles"] for u in tus)
                A.suggestions.append(Suggestion(
                    tier, rule, k, str(root), "dir", "mixed", nb, nf, why, kept, [str(root)],
                    f"{nf}:{nb}:{max(u['max_mtime_ns'] for u in tus)}", _by_class(tus)))
                n += nb
            else:
                for u in tus:
                    if u["cls"] == UNKNOWN:
                        continue
                    pl = _expand_units([u], root)
                    A.suggestions.append(Suggestion(
                        tier, rule, k, str(root / u["unit"]), "files" if u["unit"].endswith("/*") else "dir",
                        u["cls"], u["bytes"], u["nfiles"], why, kept, pl,
                        f"{u['nfiles']}:{u['bytes']}:{u['max_mtime_ns']}", {u["cls"]: u["bytes"]}))
                    n += u["bytes"]
        return n

    handled: set[str] = set()   # datasets already wholly suggested

    # ---- clusters ------------------------------------------------------------------------
    for comp in comps:
        members = sorted(k for k in comp if k in verdict)
        if len(members) < 2:
            continue
        info = {"members": members, "state": "", "keeper": "", "redundant": []}

        def rank(k):
            v = verdict[k]
            fo = final_ok.get(k, (False, ""))[0]
            st = evidence.overall_status(stages.get(k, {}), rows[k]["signal_flag"]) == "done"
            return (0 if v.rule == "P1" else 1, 0 if fo else 1, 0 if st else 1,
                    -raw_bytes.get(k, 0), _root_rank(rows[k]["leaf_dir"]), k)

        keeper = sorted(members, key=rank)[0]
        rel = {m: dupes.relation(keeper, m, raw, digest_of) for m in members if m != keeper}
        A.pending_files += sum(r.pending for r in rel.values())
        redundant = [m for m, r in rel.items() if r.b_in_a]
        unresolved = [m for m, r in rel.items() if not r.b_in_a]
        info.update(keeper=keeper, redundant=redundant, unresolved=unresolved,
                    pending=sum(r.pending for r in rel.values()))
        A.cluster_info.append(info)

        # a human delete on one copy is inherited by copies proven to be subsets of it
        d1 = [m for m in members if verdict[m].rule == "D1"]
        for m in d1:
            for o in members:
                if o == m or o in handled or verdict[o].rule == "D1":
                    continue
                r_mo = dupes.relation(m, o, raw, digest_of)
                if r_mo.b_in_a and verdict[o].rule in ("K0",):
                    verdict[o] = Verdict("D1", SUGGEST_DELETE, [f"byte-identical subset of {m} which is labelled delete"])
                elif verdict[o].rule == "K0" and (r_mo.pending or not r_mo.b_in_a):
                    verdict[o] = Verdict("R1", REVIEW, [f"possible copy of {m} (labelled delete) but identity not proven: "
                                                       f"{r_mo.pending} file(s) unhashed, {r_mo.unmatched_b} differ"])
        if d1:
            continue

        kv = verdict[keeper]
        if kv.action != KEEP:
            continue
        for m in redundant:
            if verdict[m].rule not in ("K0",):
                A.notes.append(f"{m}: redundant copy of {keeper} but verdict {verdict[m].rule}; left alone")
                continue
            kf = final_ok.get(keeper, (False, ""))[0]
            mf_any = bool(conn.execute("SELECT 1 FROM files WHERE dataset_key=? AND cls='final' LIMIT 1", (m,)).fetchone())
            if mf_any and not kf:
                A.notes.append(f"{m}: redundant raw copy of {keeper}, held because {keeper}'s final output is not verified complete")
                continue
            whole_copy(m, "A1", "DUP1", f"every raw file byte-identical (SHA-256) to {keeper}; its outputs are redundant with {keeper}'s verified final", keeper)
            handled.add(m)

    # ---- per-dataset verdict actions -----------------------------------------------------
    for k in keys:
        v = verdict[k]
        if k in handled:
            continue
        if v.rule == "D1":
            whole_copy(k, "B", "D1", "; ".join(v.reasons))
            handled.add(k)
        elif v.rule == "R1":
            A.review.append(ReviewItem(k, total_bytes.get(k, 0), v.reasons))
        elif v.rule in ("K0", "P1"):
            ok, detail = final_ok.get(k, (False, "final output not checked"))
            if not ok:
                continue
            r = rows[k]
            for u in units.get(k, []):
                if u["cls"] not in (INTER, VIEWER) or u["bytes"] < MIN_SUGGEST_BYTES:
                    continue
                name = u["unit"].rsplit("/", 1)[-1]
                if name == "DSR_decon" or u["unit"].endswith("DSR_decon"):
                    A.notes.append(f"{k}: {u['unit']} is not the registry's final but looks like a decon result; held")
                    continue
                root = _tree_root(u["tree"], r["leaf_dir"], mirror_root, None)
                tier, rule = ("A3", "I1b") if u["cls"] == VIEWER else ("A2", "I1a")
                A.suggestions.append(Suggestion(
                    tier, rule, k, str(root / u["unit"]), "files" if u["unit"].endswith("/*") else "dir",
                    u["cls"], u["bytes"], u["nfiles"],
                    f"regenerable {u['cls']}; final output verified: {detail}",
                    "", _expand_units([u], root), f"{u['nfiles']}:{u['bytes']}:{u['max_mtime_ns']}",
                    {u["cls"]: u["bytes"]}))

    suggested_roots = {s.dataset_key: 0 for s in A.suggestions}
    for s in A.suggestions:
        suggested_roots[s.dataset_key] += s.nbytes
    for k in keys:
        A.kept_bytes[k] = max(0, total_bytes.get(k, 0) - suggested_roots.get(k, 0))
    return A
