"""`[backfill]` log lines: timestamped, and per-dataset notices said once.

A --watch pass runs every 120 s, and before 2026-10-25 it printed the same ~140
lines every time (63 of them one error each for stores that never had pixel data),
none timestamped: about 17 MB a day in which a real failure could not be found or dated.
A notice keyed to a dataset is now printed again only when its text changes. The
registry's `error_message` stays the record; the log says what is new.

What was said is kept in small files under the jobs dir (tmpfs), because the crop
workers that hit these errors are separate processes, new every pass.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

from opym import lanes

_SEEN_ENV = "OPYM_BACKFILL_LOG_SEEN_DIR"


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [backfill] {msg}", flush=True)


def _seen_dir() -> Path:
    return Path(os.environ.get(_SEEN_ENV) or lanes.jobs_dir() / "log_seen")


def log_once(key: str, msg: str) -> bool:
    """`log(msg)` unless the last notice for `key` said exactly this. Returns
    whether it was printed."""
    path = _seen_dir() / hashlib.sha1(key.encode()).hexdigest()
    try:
        if path.read_text() == msg:
            return False
    except OSError:
        pass
    log(msg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(msg)
        os.replace(tmp, path)
    except OSError:
        pass  # saying it again next pass is the worst case
    return True


def forget(key: str) -> None:
    """The notice for `key` no longer applies (e.g. the dataset now processed):
    the next one is printed even if it repeats an old text."""
    try:
        (_seen_dir() / hashlib.sha1(key.encode()).hexdigest()).unlink()
    except OSError:
        pass
