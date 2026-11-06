"""The test suite must be unable to reach production state (docs/ROADMAP.md item 1).

The guards live in tests/conftest.py. These tests are their negative control: each one
fails if the matching guard is removed. Nothing here touches the real /dev/shm/petakit_jobs:
the guard test points the guard at a fake "production" dir under tmp_path (so removing the
guard creates files only in tmp), and a separate pure test pins the real constant.
"""

import os
from pathlib import Path

import conftest  # tests/conftest.py (pytest's default prepend import mode)
import pytest

PRODUCTION_REGISTRY = "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3"


def test_registry_default_is_not_production():
    from backfill import cli

    path = str(cli.DEFAULT_REGISTRY_PATH)
    assert path != PRODUCTION_REGISTRY
    assert "bioimaging-tests-" in path
    assert not path.startswith("/mmfs")


def test_petakit_queue_dir_is_not_under_dev_shm(tmp_path):
    from opym import petakit

    assert not str(petakit.QUEUE_DIR).startswith("/dev/shm")
    assert Path(petakit.QUEUE_DIR).is_relative_to(tmp_path)
    # the default arguments that captured QUEUE_DIR at def time follow it
    assert petakit._ensure_directories.__defaults__ == (petakit.QUEUE_DIR,)
    petakit._ensure_directories()
    assert Path(petakit.QUEUE_DIR).is_dir()


def test_default_submit_queue_is_the_tmp_queue(tmp_path):
    """A submit that forgets queue_dir= must land in the tmp queue, not production's."""
    import inspect

    from opym import petakit

    submits = {
        n: inspect.signature(f).parameters["queue_dir"].default
        for n, f in vars(petakit).items()
        if n.startswith("submit_") and inspect.isfunction(f) and "queue_dir" in inspect.signature(f).parameters
        and inspect.signature(f).parameters["queue_dir"].default is not inspect.Parameter.empty
    }
    assert len(submits) >= 3, submits
    for name, default in submits.items():
        assert Path(default).is_relative_to(tmp_path), name


def test_production_jobs_dir_constant_and_matcher():
    """Pure: pins the real path the guard protects. Runs before any patching, no filesystem writes."""
    assert conftest.PRODUCTION_JOBS_DIR == "/dev/shm/petakit_jobs"
    assert conftest._is_production_path("/dev/shm/petakit_jobs/queue/x.json") is True
    assert conftest._is_production_path("/dev/shm/other") is False


def test_guard_blocks_production_jobs_dir(tmp_path, monkeypatch):
    # The guard reads PRODUCTION_JOBS_DIR at call time: aim it at a fake dir so that, if the
    # guard were ever removed, this test would create files in tmp_path and not in the live queue.
    fake_prod = tmp_path / "fake_prod" / "petakit_jobs"
    monkeypatch.setattr(conftest, "PRODUCTION_JOBS_DIR", str(fake_prod))
    probe_dir = fake_prod / "__guard_probe__"
    probe_file = fake_prod / "queue" / "__guard_probe__.json"

    with pytest.raises(AssertionError, match="test touched production"):
        probe_dir.mkdir(parents=True)
    with pytest.raises(AssertionError, match="test touched production"):
        os.makedirs(probe_dir)
    with pytest.raises(AssertionError, match="test touched production"):
        open(probe_file, "w")
    with pytest.raises(AssertionError, match="test touched production"):
        probe_file.write_text("{}")
    with pytest.raises(AssertionError, match="test touched production"):
        os.replace(probe_file, probe_file.with_suffix(".done"))

    assert not fake_prod.exists()


def test_guard_allows_everything_else(tmp_path):
    target = tmp_path / "queue" / "x.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    assert target.read_text() == "{}"
