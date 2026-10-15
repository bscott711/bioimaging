"""`storage_triage.sqlite3`: this tool's own inventory + hash cache."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    dataset_key TEXT PRIMARY KEY,
    scanned_at TEXT NOT NULL,
    final_rel TEXT,
    final_tree TEXT
);
-- Per-file rows only for the classes we hash or verify (raw, final).
CREATE TABLE IF NOT EXISTS files (
    dataset_key TEXT NOT NULL,
    tree TEXT NOT NULL,         -- leaf | mirror | master
    rel TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    cls TEXT NOT NULL,
    PRIMARY KEY (dataset_key, tree, rel)
);
CREATE INDEX IF NOT EXISTS files_size ON files(size);
-- Everything else is aggregated into deletable units (a dir, or the loose
-- files of one dir).
CREATE TABLE IF NOT EXISTS units (
    dataset_key TEXT NOT NULL,
    tree TEXT NOT NULL,
    unit TEXT NOT NULL,         -- relpath; trailing "/*" = loose files of that dir
    cls TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    nfiles INTEGER NOT NULL,
    max_mtime_ns INTEGER NOT NULL,
    PRIMARY KEY (dataset_key, tree, unit, cls)
);
CREATE TABLE IF NOT EXISTS final_checks (
    dataset_key TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,   -- nfiles:bytes:max_mtime of the final output when checked
    ok INTEGER NOT NULL,
    detail TEXT NOT NULL,
    checked_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hashes (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    hashed_at TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def connect_ro(path: Path) -> sqlite3.Connection | None:
    """Read-only handle on one of the *other* databases (registry, triage, ...),
    or None when it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn
