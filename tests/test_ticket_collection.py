"""Every submitted deskew ticket is collected, and a dataset only ever looks
finished when its output was made with the current settings.

Rig 2026-09-24: 428 decon tickets submitted on 9/20 finished within days, but
no pass ever collected them -- watch-mode passes don't wait, and the next
pass's Phase A never handed them back because `detect_rois` short-circuits on
a (stale) `mip_encode: done`. opym-dashboard showed all of them as done, with
their pre-decon movies.

Real `StatusRegistry` on tmp_path; no MATLAB, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from opym.decon_config import decon_params_fingerprint
from opym.discovery import KIND_ZARR_PRECROPPED
from opym.registry import StatusRegistry

from backfill import cli, pipeline

CURRENT_PARAMS = {
    "run_decon": True,
    "wiener_alpha": 0.2,
    "otf_cum_thresh": 0.9,
    "hann_win_bounds": [0.4, 1.0],
    "damp_factor": 2.0,  # tickets store floats; the fingerprint constant is int 2
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.delenv("OPYM_DECON_REPROCESS_LEGACY", raising=False)
    jobs = tmp_path / "petakit_jobs"
    for d in ("queue", "queue_live", "completed", "failed"):
        (jobs / d).mkdir(parents=True)
    registry = StatusRegistry(tmp_path / "registry.sqlite3")
    encoded = []
    monkeypatch.setattr(cli, "_run_mip_encode", lambda ds, reg, fps: encoded.append(ds.dataset_key))
    monkeypatch.setattr(cli, "_dsr_dir_for", lambda ds: tmp_path / "DSR_decon")
    yield SimpleNamespace(
        tmp=tmp_path, psf=str(psf.resolve()), jobs=jobs, registry=registry, encoded=encoded
    )
    registry.close()


def _dataset(env, key="Cell_1"):
    """A dataset whose 8/05 pass finished (stale MIPs) and whose deskew was
    re-submitted since, with its ticket queued under `queue/`."""
    reg = env.registry
    reg.register_dataset(key, root=str(env.tmp), leaf_dir=str(env.tmp / key), master_file="m")
    reg.finish_stage(key, "roi_detect", status="done")
    reg.finish_stage(key, "mip_encode", status="done", output_path="/old/mip_movies")
    ticket = env.jobs / "queue" / f"DESKEW_{key}.json"
    reg.start_stage(key, "deskew", ticket_path=str(ticket))
    return SimpleNamespace(dataset_key=key, kind="tiff"), ticket


def _finish(env, ticket, params, where="completed"):
    (env.jobs / where / ticket.name).write_text(
        json.dumps({"jobType": "deskew", "parameters": {"psf_path": env.psf, **params}})
    )


def _collect(env, ds):
    pending = cli._inflight_tickets(env.registry, {ds.dataset_key: ds})
    cli._drain_resolved_tickets(pending, {ds.dataset_key: ds}, env.registry, 12.0)
    return pending


def test_a_ticket_finished_in_an_earlier_pass_is_collected(env):
    ds, ticket = _dataset(env)
    _finish(env, ticket, CURRENT_PARAMS)

    assert not _collect(env, ds)
    assert env.registry.is_stage_done(ds.dataset_key, "deskew")
    assert env.encoded == [ds.dataset_key]


def test_a_ticket_made_with_old_settings_is_requeued_not_shown(env):
    """The 9/20 batch: finished with wiener_alpha 0.02, before the other
    knobs were set explicitly."""
    ds, ticket = _dataset(env)
    _finish(env, ticket, {"run_decon": True, "wiener_alpha": 0.02})

    _collect(env, ds)

    reg = env.registry
    assert env.encoded == [], "encoded MIPs of output that is about to be replaced"
    assert reg.get_decon_params(ds.dataset_key) == "a0.02_o?_h?_d?"  # honest, not NULL
    assert reg.get_stage(ds.dataset_key, "mip_encode")["status"] == "pending"
    assert reg.get_stage(ds.dataset_key, "mip_encode")["output_path"] == "/old/mip_movies"
    # ... so the next pass's submit sees a mismatch and re-submits.
    assert not pipeline.decon_provenance_matches(reg, ds.dataset_key, Path(env.psf))


def test_a_failed_ticket_is_recorded_as_failed(env):
    ds, ticket = _dataset(env)
    _finish(env, ticket, CURRENT_PARAMS, where="failed")

    _collect(env, ds)

    assert env.registry.get_stage(ds.dataset_key, "deskew")["status"] == "failed"
    assert env.encoded == []


@pytest.mark.parametrize("spelling", ["{}", ".active_{}", ".requeue_{}"])
def test_a_queued_or_claimed_ticket_stays_in_flight(env, spelling):
    ds, ticket = _dataset(env)
    (env.jobs / "queue" / spelling.format(ticket.name)).write_text("{}")

    assert ds.dataset_key in _collect(env, ds)
    assert env.registry.get_stage(ds.dataset_key, "deskew")["status"] == "running"


def test_a_ticket_in_no_directory_is_marked_lost(env):
    ds, _ticket = _dataset(env)

    assert not _collect(env, ds)
    row = env.registry.get_stage(ds.dataset_key, "deskew")
    assert row["status"] == "failed" and "ticket lost" in row["error_message"]


def test_resubmitting_resets_mip_encode_and_skips_roi_detection(env, monkeypatch):
    """An already-cropped TIFF dataset goes straight to the deskew submit --
    `detect_rois`' mip_encode gate used to hide it -- and the submit marks
    its old MIPs as out of date."""
    reg = env.registry
    master = env.tmp / "Cell_1" / "cell_MMStack_Pos0.ome.tif"
    master.parent.mkdir()
    master.write_bytes(b"raw")
    ds = SimpleNamespace(
        dataset_key="Cell_1", kind="tiff", root=env.tmp, leaf_dir=master.parent,
        raw_dir=master.parent, master_file=master, channel_zarr_paths=(),
    )
    tiff_out = pipeline.derive_paths(master, pipeline.OutputFormat.TIFF_SERIES).output_dir
    tiff_out.mkdir(parents=True)
    (tiff_out / "cell_MMStack_Pos0_C0_T000.tif").write_bytes(b"cropped")
    reg.register_dataset(
        "Cell_1", root=str(env.tmp), leaf_dir=str(master.parent), master_file=str(master)
    )
    reg.finish_stage("Cell_1", "crop_tiff", status="done")
    reg.finish_stage("Cell_1", "deskew", status="done")
    reg.finish_stage("Cell_1", "mip_encode", status="done", output_path="/old/mip_movies")
    reg.set_decon_psf("Cell_1", env.psf)
    reg.set_decon_params("Cell_1", "a0.02_o?_h?_d?")

    monkeypatch.setattr(pipeline, "detect_rois", lambda *a: pytest.fail("re-ran ROI detection"))
    ticket = env.jobs / "queue" / "new.json"
    monkeypatch.setattr(pipeline, "_admitted_submit", lambda **kw: ticket)

    assert pipeline.process_crop_and_submit(ds, reg) == ticket
    assert reg.get_stage("Cell_1", "deskew")["status"] == "running"
    assert reg.get_decon_params("Cell_1") == decon_params_fingerprint()
    assert reg.get_stage("Cell_1", "mip_encode")["status"] == "pending"


def test_triage_runs_once_per_dataset(env):
    reg = env.registry
    fresh = SimpleNamespace(dataset_key="fresh", kind="tiff")
    triaged, _ = _dataset(env, "triaged")
    reg.register_dataset("fresh", root="r", leaf_dir="l", master_file="m")
    reg.reset_stage(triaged.dataset_key, "mip_encode")  # re-queued: not done any more

    assert cli._needs_triage(fresh, reg)
    assert not cli._needs_triage(triaged, reg)
    assert not cli._needs_triage(SimpleNamespace(dataset_key="z", kind=KIND_ZARR_PRECROPPED), reg)


def test_declared_timepoints_expose_an_aborted_acquisition(tmp_path):
    """Cell_004, 2026-09-24: configured for 100 timepoints, stopped after 2."""
    stores = []
    for ch in ("GFP_488", "mScarlet_561"):
        store = tmp_path / f"Cell_004_{ch}.ome.zarr"
        zarr.open_group(str(store), mode="w")
        arr = zarr.open(
            str(store / "p0"), mode="w", shape=(100, 2, 3, 4), chunks=(1, 1, 3, 4),
            dtype="uint16", dimension_separator="/",
        )
        arr[0] = np.ones((2, 3, 4), dtype="uint16")
        arr[1] = np.ones((2, 3, 4), dtype="uint16")
        stores.append(store)
    ds = SimpleNamespace(channel_zarr_paths=tuple(stores))

    assert pipeline.dataset_declared_timepoints(ds) == 100
    assert pipeline.dataset_timepoints(ds) == 2
