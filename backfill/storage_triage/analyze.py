"""Turn inventory + hashes + evidence into suggestions."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from opym.store import DSR_STORE_SUFFIX, read_progress

from backfill.storage_triage import dupes, evidence
from backfill.storage_triage.completeness import CHECK_VERSION, verify_final
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
    scan_times: dict[str, str] = field(default_factory=dict)       # dataset_key -> scanned_at (ISO)


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


def scan_ages(
    conn: sqlite3.Connection, keys: list[str], now: datetime | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Age in hours of each dataset's inventory row (`scans.scanned_at`), and the keys
    registered but never scanned. Suggestions are only as fresh as this cache."""
    now = now or datetime.now(UTC)
    want = set(keys)
    ages: dict[str, float] = {}
    for k, ts in conn.execute("SELECT dataset_key, scanned_at FROM scans"):
        if k not in want:
            continue
        try:
            t = datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        ages[k] = (now - t).total_seconds() / 3600
    return ages, sorted(want - set(ages))


_FINAL_FRAME = re.compile(r"_C(\d+)_T(\d+)\.tif$")


def final_tiff_channels(conn: sqlite3.Connection, key: str) -> set[int]:
    """Channels for which the dataset's final dir holds `*_C#_T#.tif` frames."""
    out: set[int] = set()
    for (rel,) in conn.execute("SELECT rel FROM files WHERE dataset_key=? AND cls='final'", (key,)):
        m = _FINAL_FRAME.search(rel.rsplit("/", 1)[-1])
        if m:
            out.add(int(m.group(1)))
    return out


def _zarr_channel_count(store: Path) -> int | None:
    """Channel count from the zarr's own metadata when `.opym_live.json` is absent (backfill
    exports): `omero.channels` in `.zattrs` (v2) or `zarr.json` (v3, bare or under `ome`).
    Conservative: None unless it is a positive count, and, where the first array's shape and
    the multiscales axes are readable, the channel axis must agree with it."""
    attrs = None
    for name in (".zattrs", "zarr.json"):
        try:
            doc = json.loads((store / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            attrs = doc.get("attributes", doc) if name == "zarr.json" else doc
            break
    if not isinstance(attrs, dict):
        return None
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    chans = (ome.get("omero") or {}).get("channels") if isinstance(ome.get("omero"), dict) else None
    if not isinstance(chans, list) or not chans:
        return None
    n = len(chans)
    try:
        axes = [a.get("name") if isinstance(a, dict) else a for a in ome["multiscales"][0]["axes"]]
        shape = json.loads((store / "0" / ".zarray").read_text())["shape"]
        if "c" in axes and len(axes) == len(shape) and shape[axes.index("c")] != n:
            return None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass                      # axes or shape unreadable: the omero list alone stands
    return n


def viewer_hold_reason(unit: Path, tiff_channels: set[int]) -> str | None:
    """Why a VIEWER unit (the `viewer/` dir, or one `*_dsr.ome.zarr` in it) must not be
    offered, or None. The TIFF fallback is not equivalent (cell-cinema and ChimeraX-NGFF
    read only the zarr), and with output_format=ome-zarr the zarr is the only DSR copy:
    offer it only when TIFF frames exist for every channel of every zarr it holds."""
    try:
        stores = [unit] if unit.name.endswith(DSR_STORE_SUFFIX) else sorted(
            d for d in unit.iterdir() if d.name.endswith(DSR_STORE_SUFFIX)
        )
    except OSError as exc:
        return f"viewer directory unreadable ({exc.strerror or exc})"
    if not stores:
        return "no viewer zarr store found to check"
    for store in stores:
        meta = read_progress(store)
        n_c = meta.get("n_c") if isinstance(meta, dict) else None
        if not isinstance(n_c, int) or n_c < 1:
            n_c = _zarr_channel_count(store)      # backfill exports carry no .opym_live.json
        if not isinstance(n_c, int) or n_c < 1:
            return f"channel count of {store.name} is unknown, so its TIFF fallback cannot be checked"
        if not tiff_channels:
            return "the final output has no TIFF frames, so the viewer zarr may be the only DSR copy"
        lacking = sorted(set(range(n_c)) - tiff_channels)
        if lacking:
            return f"the final output has no TIFF frames for channel(s) {lacking} of {store.name} ({n_c} channels)"
    return None


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
        fp = f"v{CHECK_VERSION}:{agg[0]}:{agg[1]}:{agg[2]}"
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
    A.scan_times = {r[0]: r[1] for r in conn.execute("SELECT dataset_key, scanned_at FROM scans")}
    scanned = set(A.scan_times)

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
        r = rows.get(f.dataset_key)
        if r is None:      # retired from the registry: its files prove nothing
            return None
        p = f.abspath(r["leaf_dir"], mirror_root, r["master_file"])
        return hashes.get((str(p), f.size, f.mtime_ns))

    comp_of: dict[str, int] = {}
    comps = dupes.clusters(pairs)
    for i, c in enumerate(comps):
        for k in c:
            comp_of[k] = i

    # any dataset sharing a raw file (even a small partial copy) can carry a conflicting label
    partners: dict[str, set[str]] = defaultdict(set)
    for a_, b_ in dupes.sharing_pairs(raw, digest_of):
        partners[a_].add(b_)
        partners[b_].add(a_)

    def facts_for(k: str) -> Facts:
        r = rows[k]
        sibs = (set(comps[comp_of[k]]) if k in comp_of else set()) | partners.get(k, set())
        sib = {s: labels[s] for s in sorted(sibs) if s != k and s in labels}
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
                # one row per channel store of the sample (all of them are raw and hashed)
                by_store: dict[str, list] = defaultdict(list)
                for f in raw.get(k, []):
                    if f.tree == "master":
                        by_store[f.rel.split("/", 1)[0]].append(f)
                if not by_store:
                    continue
                base = _tree_root("master", r["leaf_dir"], mirror_root, r["master_file"]).parent
                for name, mf in sorted(by_store.items()):
                    root = base / name
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

    def files_of(k: str) -> list[tuple[dupes.RawFile, Path]]:
        r = rows[k]
        return [(f, f.abspath(r["leaf_dir"], mirror_root, r["master_file"])) for f in raw.get(k, [])]

    def roots_of(k: str) -> list[Path]:
        r = rows[k]
        out = [x for x in (_tree_root("leaf", r["leaf_dir"], mirror_root, None),
                           _tree_root("mirror", r["leaf_dir"], mirror_root, None)) if x.is_dir()]
        if r["master_file"] and any(f.tree == "master" for f in raw.get(k, [])):
            parent = _tree_root("master", r["leaf_dir"], mirror_root, r["master_file"]).parent
            out += [parent / n for n in sorted({f.rel.split("/", 1)[0] for f in raw[k] if f.tree == "master"})]
        return out

    def copy_problem(keeper: str, m: str) -> str | None:
        """Re-check, at report time, that the kept copy still exists unchanged (os.stat) and
        is not the same storage as the copy offered for deletion."""
        mf = files_of(m)
        tw = dupes.twin_files(files_of(keeper), mf, digest_of)
        bad = dupes.keeper_file_problems(tw)
        if bad:
            return bad[0] + (f" (+{len(bad) - 1} more)" if len(bad) > 1 else "")
        return dupes.alias_problem(tw, mf, roots_of(keeper), roots_of(m))

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
            problem = copy_problem(keeper, m)
            if problem:
                A.notes.append(f"{m}: byte-identical to {keeper} by hash but held: {problem}")
                continue
            kf = final_ok.get(keeper, (False, ""))[0]
            mf_any = bool(conn.execute("SELECT 1 FROM files WHERE dataset_key=? AND cls='final' LIMIT 1", (m,)).fetchone())
            if mf_any and not kf:
                A.notes.append(f"{m}: redundant raw copy of {keeper}, held because {keeper}'s final output is not verified complete")
                continue
            whole_copy(m, "A1", "DUP1", f"every raw file byte-identical (SHA-256) to {keeper}; its outputs are redundant with {keeper}'s verified final", keeper)
            handled.add(m)

    # ---- per-dataset verdict actions -----------------------------------------------------
    n_held = 0
    for k in keys:
        v = verdict[k]
        if k in handled:
            continue
        if v.rule == "D1":
            whole_copy(k, "B", "D1", "; ".join(v.reasons))
            handled.add(k)
        elif v.rule == "R1":
            A.review.append(ReviewItem(k, total_bytes.get(k, 0), v.reasons))
        elif v.rule == "P1":
            # protected datasets (seglab training cells, PSF sources, keep labels, pins) keep
            # their intermediates and viewer copies too
            n_held += 1
        elif v.rule == "K0":
            ok, detail = final_ok.get(k, (False, "final output not checked"))
            if not ok:
                continue
            r = rows[k]
            tiff_channels = final_tiff_channels(conn, k)
            for u in units.get(k, []):
                if u["cls"] not in (INTER, VIEWER) or u["bytes"] < MIN_SUGGEST_BYTES:
                    continue
                name = u["unit"].rsplit("/", 1)[-1]
                if name == "DSR_decon" or u["unit"].endswith("DSR_decon"):
                    A.notes.append(f"{k}: {u['unit']} is not the registry's final but looks like a decon result; held")
                    continue
                root = _tree_root(u["tree"], r["leaf_dir"], mirror_root, None)
                if u["cls"] == VIEWER:
                    why_held = viewer_hold_reason(root / u["unit"], tiff_channels)
                    if why_held:
                        A.notes.append(f"{k}: {u['unit']} held: {why_held}")
                        continue
                tier, rule = ("A3", "I1b") if u["cls"] == VIEWER else ("A2", "I1a")
                A.suggestions.append(Suggestion(
                    tier, rule, k, str(root / u["unit"]), "files" if u["unit"].endswith("/*") else "dir",
                    u["cls"], u["bytes"], u["nfiles"],
                    f"regenerable {u['cls']}; final output verified: {detail}",
                    "", _expand_units([u], root), f"{u['nfiles']}:{u['bytes']}:{u['max_mtime_ns']}",
                    {u["cls"]: u["bytes"]}))

    if n_held:
        A.notes.append(
            f"{n_held} protected (P1) dataset(s): P1 for any reason (keep label, sibling keep, seglab, PSF source, pin) "
            "withholds intermediates and viewer copies (A2/A3)"
        )
    suggested_roots = {s.dataset_key: 0 for s in A.suggestions}
    for s in A.suggestions:
        suggested_roots[s.dataset_key] += s.nbytes
    for k in keys:
        A.kept_bytes[k] = max(0, total_bytes.get(k, 0) - suggested_roots.get(k, 0))
    return A
