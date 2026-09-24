"""Watch-mode must keep discovering new data while tickets are in flight.

Rig 2026-09-23: the watcher's Sep 21 pass waited on 489 in-flight decon
tickets (eight of them orphaned by a restarted PetaKit5D server, so they could
never resolve), and no dataset was discovered for two days after.
"""

import json
from pathlib import Path

import pytest

from backfill import cli


def test_finish_pending_does_not_wait_in_watch_mode(monkeypatch):
    drained = []
    monkeypatch.setattr(cli, "_drain_resolved_tickets", lambda p, *a: drained.append(dict(p)))
    monkeypatch.setattr(cli.time, "sleep", lambda s: pytest.fail("watch pass must not sleep"))

    pending = {"never-resolves": Path("/dev/shm/petakit_jobs/queue/x.json")}
    cli._finish_pending(pending, {}, None, 12.0, wait=False, poll_interval_s=30.0)

    assert len(drained) == 1
    assert "never-resolves" in pending      # left for the next pass to re-collect


def test_finish_pending_waits_in_one_shot_mode(monkeypatch):
    pending = {"a": Path("a.json")}
    calls = []

    def drain(p, *a):
        calls.append(1)
        if len(calls) == 3:
            p.clear()

    monkeypatch.setattr(cli, "_drain_resolved_tickets", drain)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    cli._finish_pending(pending, {}, None, 12.0, wait=True, poll_interval_s=0.0)
    assert calls == [1, 1, 1] and not pending


def test_watch_backfill_passes_do_not_wait(monkeypatch):
    seen = {}

    def fake_run(roots, **kwargs):
        seen.update(kwargs)
        raise KeyboardInterrupt  # end the infinite loop after one pass

    monkeypatch.setattr(cli, "run_backfill", fake_run)
    with pytest.raises(KeyboardInterrupt):
        cli.watch_backfill([Path("/nowhere")])
    assert seen["wait_for_pending"] is False


class _Ds:
    def __init__(self, stores):
        self.channel_zarr_paths = stores


def _store(tmp_path, name, fmt=None):
    p = tmp_path / name
    p.mkdir()
    attrs = {"multiscales": []}
    if fmt:
        attrs["opym"] = {"output_format": fmt}
    (p / ".zattrs").write_text(json.dumps(attrs))
    return p


def test_output_format_prefers_store_then_env_then_both(tmp_path, monkeypatch):
    monkeypatch.delenv("OPYM_OUTPUT_FORMAT", raising=False)
    plain = _store(tmp_path, "a_GFP_488.ome.zarr")
    assert cli._resolve_output_format(_Ds([plain])) == "both"

    monkeypatch.setenv("OPYM_OUTPUT_FORMAT", "tiff")
    assert cli._resolve_output_format(_Ds([plain])) == "tiff"

    chosen = _store(tmp_path, "b_GFP_488.ome.zarr", fmt="ome-zarr")
    assert cli._resolve_output_format(_Ds([chosen])) == "ome-zarr"

    monkeypatch.setenv("OPYM_OUTPUT_FORMAT", "bogus")
    assert cli._resolve_output_format(_Ds([plain])) == "both"
    assert cli._resolve_output_format(_Ds(None)) == "both"


def test_watch_backfill_pauses_while_a_live_lease_is_held(monkeypatch):
    from opym import lanes

    lanes.write_lease(["live-session"])
    events = []

    def fake_sleep(s):
        events.append(("sleep", s))
        lanes.release_lease()  # the acquisition ends during the pause

    def fake_run(roots, **kwargs):
        events.append(("pass",))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    monkeypatch.setattr(cli, "run_backfill", fake_run)
    with pytest.raises(KeyboardInterrupt):
        cli.watch_backfill([Path("/nowhere")])
    assert events == [("sleep", cli._LEASE_POLL_S), ("pass",)]
