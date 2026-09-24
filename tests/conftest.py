import pytest


@pytest.fixture(autouse=True)
def _isolate_petakit_jobs_dir(tmp_path, monkeypatch):
    """opym.lanes (backfill admission, the live lease) defaults to production's
    /dev/shm/petakit_jobs. Tests must never read a real lease there or create
    files in it."""
    monkeypatch.setenv("PETAKIT_JOBS_DIR", str(tmp_path / "petakit_jobs"))
