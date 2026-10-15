import pytest


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
