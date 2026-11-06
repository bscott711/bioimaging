import atexit
import builtins
import io
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# --- Session isolation: runs before any bioimaging or opym module is imported. ---
# `backfill.cli.DEFAULT_REGISTRY_PATH` is evaluated at import from this variable, so the
# production registry (whatever the unit or the shell set) must be replaced first. A
# fixture is too late for it. Always override, never setdefault.
_SESSION_TMP = tempfile.mkdtemp(prefix="bioimaging-tests-")
atexit.register(shutil.rmtree, _SESSION_TMP, ignore_errors=True)
os.environ["OPYM_BACKFILL_REGISTRY_PATH"] = os.path.join(_SESSION_TMP, "registry.sqlite3")
os.environ["PETAKIT_JOBS_DIR"] = os.path.join(_SESSION_TMP, "petakit_jobs")

import backfill.cli  # noqa: E402  (must come after the environment is set)

if not str(backfill.cli.DEFAULT_REGISTRY_PATH).startswith(_SESSION_TMP):
    raise pytest.UsageError(
        f"backfill.cli.DEFAULT_REGISTRY_PATH is {backfill.cli.DEFAULT_REGISTRY_PATH}, not under the "
        f"test tmp dir {_SESSION_TMP}: backfill.cli was imported before tests/conftest.py ran "
        "(a plugin, -p option, or another conftest). Refusing to run against a real registry."
    )

# --- Production jobs dir guard -----------------------------------------------------
# `opym.petakit` uses a constant `QUEUE_DIR = /dev/shm/petakit_jobs/queue` (also as a default
# argument of every submit_* function) that ignores PETAKIT_JOBS_DIR. A test that forgets to
# patch a submit would queue a job for the real MATLAB servers. Two layers:
#   1. QUEUE_DIR and the defaults that captured it are redirected to a per-test tmp dir.
#   2. Any open/mkdir/replace under PRODUCTION_JOBS_DIR raises, whoever asks. The guard reads
#      this constant at call time (tests/test_test_safety_guards.py relies on that).
PRODUCTION_JOBS_DIR = "/dev/shm/petakit_jobs"


def _is_production_path(path) -> bool:
    try:
        resolved = os.path.realpath(os.fspath(path))
    except (TypeError, ValueError):
        return False  # file descriptors and other non-paths are not ours to judge
    prod = os.path.realpath(PRODUCTION_JOBS_DIR)
    return resolved == prod or resolved.startswith(prod + os.sep)


_PATH_KWARGS = ("file", "path", "src", "dst")


def _guarded(real, what):
    def wrapper(*args, **kwargs):
        # first two positionals cover open(file), replace(src, dst); keywords cover open(file=...)
        for candidate in (*args[:2], *(kwargs[k] for k in _PATH_KWARGS if k in kwargs)):
            if _is_production_path(candidate):
                raise AssertionError(
                    f"test touched production {PRODUCTION_JOBS_DIR}: {what}({os.fspath(candidate)!r}). "
                    "Pass a tmp_path (queue_dir=..., PETAKIT_JOBS_DIR) instead."
                )
        return real(*args, **kwargs)

    wrapper.__wrapped__ = real
    return wrapper


@pytest.fixture(scope="session")
def production_jobs_dir() -> str:
    """The path the guard protects (read at call time, so a proof can point it at a tmp dir)."""
    return PRODUCTION_JOBS_DIR


@pytest.fixture(autouse=True)
def _forbid_production_jobs_dir(monkeypatch):
    # Path.mkdir -> os.mkdir; Path.open/write_text/write_bytes -> io.open (same object as
    # builtins.open, but patched by name in both places); atomic ticket writes end in os.replace.
    for owner, name in (
        (os, "mkdir"), (os, "makedirs"), (os, "open"), (os, "replace"), (os, "rename"),
        (io, "open"), (builtins, "open"),
    ):
        monkeypatch.setattr(owner, name, _guarded(getattr(owner, name), f"{owner.__name__}.{name}"))


@pytest.fixture(autouse=True)
def _redirect_petakit_queue_dir(tmp_path, monkeypatch):
    """Point opym.petakit.QUEUE_DIR, and the default arguments that captured it at def time,
    at a tmp queue. Patching the module name alone would leave the defaults on /dev/shm."""
    import types

    from opym import petakit

    original = petakit.QUEUE_DIR
    redirected = tmp_path / "petakit_jobs" / "queue"
    monkeypatch.setattr(petakit, "QUEUE_DIR", redirected)
    for fn in vars(petakit).values():
        if not isinstance(fn, types.FunctionType) or fn.__module__ != petakit.__name__:
            continue
        if fn.__defaults__ and original in fn.__defaults__:
            monkeypatch.setattr(
                fn, "__defaults__", tuple(redirected if d == original else d for d in fn.__defaults__)
            )
        if fn.__kwdefaults__ and original in fn.__kwdefaults__.values():
            monkeypatch.setattr(
                fn, "__kwdefaults__",
                {k: (redirected if v == original else v) for k, v in fn.__kwdefaults__.items()},
            )


@pytest.fixture(autouse=True)
def _isolate_petakit_jobs_dir(tmp_path, monkeypatch):
    """opym.lanes (backfill admission, the live lease) defaults to production's
    /dev/shm/petakit_jobs. Tests must never read a real lease there or create
    files in it."""
    monkeypatch.setenv("PETAKIT_JOBS_DIR", str(tmp_path / "petakit_jobs"))


@pytest.fixture(autouse=True)
def _no_zarr_settle_window(monkeypatch):
    """Production waits 15 min after a zarr store's newest write before staging it
    (an in-flight Globus copy reads back as zeros). Tests build stores a moment
    ago, so they turn the window off; the tests that exercise it set it back."""
    monkeypatch.setenv("OPYM_ZARR_SETTLE_SECONDS", "0")
