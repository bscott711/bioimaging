"""Content-based duplicate detection for raw acquisitions.

Paths prove nothing (the old duplicates.csv grouped by relative path and
called 8 *partial* copies identical). A copy is redundant only when every raw
file it owns has a byte-identical twin (SHA-256) in the copy we keep.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MIN_LINK_BYTES = 1 * 2**20       # smaller files never link two datasets (AcqSettings.txt is everywhere)
MIN_SHARED_FRACTION = 0.5         # of the smaller dataset's raw bytes, to become a candidate pair
CHUNK = 8 * 2**20


@dataclass(frozen=True)
class RawFile:
    dataset_key: str
    tree: str
    rel: str
    size: int
    mtime_ns: int

    def abspath(self, leaf_dir: str, mirror_root: Path, master_file: str | None) -> Path:
        if self.tree == "leaf":
            return Path(leaf_dir) / self.rel
        if self.tree == "mirror":
            return mirror_root / leaf_dir.lstrip("/") / self.rel
        # master zarr store: rel starts with the store's own name
        store = Path(master_file[: master_file.index(".zarr") + 5])
        return store.parent / self.rel


def load_raw(conn: sqlite3.Connection) -> dict[str, list[RawFile]]:
    out: dict[str, list[RawFile]] = defaultdict(list)
    for r in conn.execute("SELECT dataset_key, tree, rel, size, mtime_ns FROM files WHERE cls='raw'"):
        out[r[0]].append(RawFile(*r))
    return out


def candidate_pairs(raw: dict[str, list[RawFile]]) -> set[tuple[str, str]]:
    """Pairs of datasets that share large raw files by size (renames included).
    Cheap pre-filter only; nothing is called a duplicate until hashes agree."""
    by_size: dict[int, set[str]] = defaultdict(set)
    total = {k: sum(f.size for f in v) for k, v in raw.items()}
    for k, files in raw.items():
        for f in files:
            if f.size >= MIN_LINK_BYTES:
                by_size[f.size].add(k)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for size, keys in by_size.items():
        if len(keys) < 2:
            continue
        ks = sorted(keys)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                shared[(a, b)] += size
    return {
        pair for pair, nbytes in shared.items()
        if nbytes >= MIN_SHARED_FRACTION * min(total[pair[0]], total[pair[1]])
    }


def sharing_pairs(raw: dict[str, list[RawFile]], digest_of) -> set[tuple[str, str]]:
    """Pairs of datasets that share ANY raw file >= MIN_LINK_BYTES, however small the
    overlap, PROVEN by an equal (size, sha256). Used to find labelled siblings. An unhashed
    file never links: equal size alone is not evidence (zarr chunks of one size recur
    across unrelated datasets), so a link claimed from it would put a false reason in front
    of the PI. Only candidate-pair datasets are ever hashed, so a partial copy is seen only
    when its shared file has been hashed."""
    by_key: dict[tuple[int, str], set[str]] = defaultdict(set)
    for k, files in raw.items():
        for f in files:
            if f.size >= MIN_LINK_BYTES:
                d = digest_of(f)
                if d is not None:
                    by_key[(f.size, d)].add(k)
    out: set[tuple[str, str]] = set()
    for owners in by_key.values():
        ks = sorted(owners)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                out.add((a, b))
    return out


def clusters(pairs: set[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[str, set[str]] = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    return list(groups.values())


class RateLimiter:
    """Shared bytes-per-second cap for all hashing threads (`--max-mbps`). Each
    `acquire(n)` reserves the next n bytes of the budget and sleeps until its slot."""

    def __init__(self, bytes_per_s: float, clock=time.monotonic, sleep=time.sleep):
        self.rate = float(bytes_per_s)
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = clock()

    def acquire(self, nbytes: int) -> None:
        with self._lock:
            now = self._clock()
            start = max(self._next, now)
            self._next = start + nbytes / self.rate
        if start > now:
            self._sleep(start - now)


def sha256_file(path: Path, limiter: RateLimiter | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while chunk := f.read(CHUNK):
            if limiter is not None:
                limiter.acquire(len(chunk))
            h.update(chunk)
    return h.hexdigest()


def keeper_file_problems(files: list[tuple[RawFile, Path]], stat=os.stat) -> list[str]:
    """`os.stat` every file of the copy we keep. The inventory and the hash cache are
    only claims about the past: a keeper file that is gone, or whose size or mtime moved,
    no longer proves anything about the copy that relied on it."""
    out = []
    for f, p in files:
        try:
            st = stat(p)
        except OSError as exc:
            out.append(f"kept copy's file is missing or unreadable: {p} ({exc.strerror or exc})")
            continue
        if st.st_size != f.size or st.st_mtime_ns != f.mtime_ns:
            out.append(f"kept copy's file changed since it was scanned: {p}")
    return out


def alias_problem(
    keeper_files: list[tuple[RawFile, Path]], member_files: list[tuple[RawFile, Path]],
    keeper_roots: list[Path], member_roots: list[Path], stat=os.stat,
) -> str | None:
    """Two "copies" that are really one: a symlinked or bind-mounted root, or a hardlink
    farm. Identical hashes then prove nothing, and removing the alias removes the keeper's
    data. Compare resolved roots (equal or nested) and (st_dev, st_ino) of the raw files."""
    def real(p: Path) -> Path:
        return Path(os.path.realpath(p))

    kr = [real(r) for r in keeper_roots]
    for m in (real(r) for r in member_roots):
        for k in kr:
            if m == k or m in k.parents or k in m.parents:
                return f"resolved path {m} overlaps the kept copy's {k}"
    ids = set()
    for _f, p in keeper_files:
        try:
            st = stat(p)
        except OSError:
            continue
        ids.add((st.st_dev, st.st_ino))
    for _f, p in member_files:
        try:
            st = stat(p)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) in ids:
            return f"{p} is the same file (device/inode) as one in the kept copy"
    return None


def twin_files(
    keeper: list[tuple[RawFile, Path]], member: list[tuple[RawFile, Path]], digest_of,
) -> list[tuple[RawFile, Path]]:
    """The keeper files that the member's files rely on: same (size, sha256)."""
    wanted = Counter((f.size, digest_of(f)) for f, _p in member)
    return [(f, p) for f, p in keeper if (f.size, digest_of(f)) in wanted]


def files_to_hash(raw: dict[str, list[RawFile]], pairs: set[tuple[str, str]]) -> list[RawFile]:
    """Every raw file of every candidate dataset (small siblings such as the
    metadata txt are needed to prove a copy is a *full* twin)."""
    # whole clusters, smallest first: a cluster is only provable once all of its
    # files are hashed, so finishing small ones early yields suggestions early
    comps = sorted(clusters(pairs), key=lambda c: (sum(f.size for k in c for f in raw[k]), sorted(c)))
    return [f for comp in comps for k in sorted(comp) for f in raw[k]]


def hash_missing(
    conn: sqlite3.Connection, todo: list[tuple[RawFile, Path]], workers: int = 8, log=print,
    hasher=sha256_file,
) -> int:
    """Hash files not yet cached (cache key: path+size+mtime). Resumable."""
    cached = {
        (r[0], r[1], r[2]) for r in conn.execute("SELECT path, size, mtime_ns FROM hashes")
    }
    pending = [(f, p) for f, p in todo if (str(p), f.size, f.mtime_ns) not in cached]
    total = sum(f.size for f, _ in pending)
    log(f"[hash] {len(pending)} files / {total / 2**30:.0f} GiB to hash ({len(todo) - len(pending)} cached)")
    done_bytes = 0

    def work(fp):
        f, p = fp
        try:
            st = os.stat(p)
            if st.st_size != f.size or st.st_mtime_ns != f.mtime_ns:
                return f, p, None, "changed since scan"
            return f, p, hasher(p), None
        except OSError as exc:
            return f, p, None, str(exc)

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for f, p, digest, err in ex.map(work, pending):
            n += 1
            done_bytes += f.size
            if digest:
                for attempt in range(20):   # ride out another writer (report / final checks)
                    try:
                        conn.execute(
                            "INSERT OR REPLACE INTO hashes VALUES (?,?,?,?,?)",
                            (str(p), f.size, f.mtime_ns, digest, datetime.now(UTC).isoformat()),
                        )
                        conn.commit()   # never hold the write lock across the next multi-GB read
                        break
                    except sqlite3.OperationalError as exc:
                        if "locked" not in str(exc) or attempt == 19:
                            raise
                        time.sleep(5)
            else:
                log(f"[hash] skip {p}: {err}")
            if n % 50 == 0:
                log(f"[hash] {n}/{len(pending)} files, {done_bytes / 2**30:.0f}/{total / 2**30:.0f} GiB")
    conn.commit()
    return n


def load_hashes(conn: sqlite3.Connection) -> dict[tuple[str, int, int], str]:
    return {(r[0], r[1], r[2]): r[3] for r in conn.execute("SELECT path, size, mtime_ns, sha256 FROM hashes")}


@dataclass
class Relation:
    """b_in_a: every raw file of b has a byte-identical twin in a."""
    a: str
    b: str
    b_in_a: bool
    a_in_b: bool
    pending: int            # files with no hash yet
    unmatched_b: int


def relation(
    a: str, b: str, raw: dict[str, list[RawFile]], digest_of,
) -> Relation:
    """`digest_of(RawFile) -> sha256 | None`."""
    def digests(k):
        d, missing = defaultdict(int), 0
        for f in raw[k]:
            h = digest_of(f)
            if h is None:
                missing += 1
            else:
                d[(f.size, h)] += 1
        return d, missing

    da, ma = digests(a)
    db, mb = digests(b)
    pending = ma + mb
    b_in_a = pending == 0 and all(da.get(k, 0) >= n for k, n in db.items()) and bool(db)
    a_in_b = pending == 0 and all(db.get(k, 0) >= n for k, n in da.items()) and bool(da)
    unmatched_b = sum(n - min(n, da.get(k, 0)) for k, n in db.items())
    return Relation(a, b, b_in_a, a_in_b, pending, unmatched_b)
