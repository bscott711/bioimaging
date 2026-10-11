"""2026-10-10 maintenance: the backfill's pool workers must not leak, and
datasets that keep failing the same way must stop being retried."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

from opym.registry import MAX_STAGE_FAILURES, StatusRegistry

from backfill import cli


def _square(x):
    return x * x


def test_pools_use_a_fork_server():
    pool = cli._new_pool(1)
    try:
        assert pool._mp_context.get_start_method() == "forkserver"
    finally:
        pool.shutdown(wait=True)


def test_abandoned_pool_workers_exit(tmp_path):
    """What every watch pass does: run work, then shut down without waiting.
    The workers must actually go away (forked ones hung for days)."""
    pool = cli._new_pool(2)
    assert list(pool.map(_square, range(4))) == [0, 1, 4, 9]
    pids = list(pool._processes)
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + 20
    alive = pids
    while alive and time.monotonic() < deadline:
        time.sleep(0.2)
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            try:  # reaped by the executor's manager thread, or a zombie
                state = open(f"/proc/{pid}/stat").read().split()[2]
            except OSError:
                continue
            if state != "Z":
                alive.append(pid)
    assert alive == []


def _reg_with(tmp_path, keys):
    reg = StatusRegistry(tmp_path / "registry.sqlite3")
    for k in keys:
        reg.register_dataset(k, root="/r", leaf_dir=f"/r/{k}", master_file=f"/r/{k}/m.tif")
    return reg


def test_given_up_datasets_are_skipped(tmp_path, capsys):
    reg = _reg_with(tmp_path, ["good", "hopeless", "flaky"])
    for _ in range(MAX_STAGE_FAILURES):
        reg.finish_stage("hopeless", "mip_encode", status="failed", error="No MIP TIFFs")
    reg.finish_stage("flaky", "crop_zarr", status="failed", error="once")
    datasets = [SimpleNamespace(dataset_key=k) for k in ("good", "hopeless", "flaky")]

    kept = cli._without_gave_up(datasets, reg)

    assert [d.dataset_key for d in kept] == ["good", "flaky"]
    assert "Skipping 1 dataset(s)" in capsys.readouterr().out


def test_nothing_skipped_and_nothing_printed_without_give_ups(tmp_path, capsys):
    reg = _reg_with(tmp_path, ["a"])
    datasets = [SimpleNamespace(dataset_key="a")]
    assert cli._without_gave_up(datasets, reg) == datasets
    assert capsys.readouterr().out == ""
