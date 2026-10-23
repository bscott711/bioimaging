"""backfill.live_index: a streamed session reaches the registry while it is
acquired, and its movies are made the moment it completes, before any raw
data has drained to GPFS."""

import json
import time

import numpy as np
import pytest
from opym import ome_zarr_writer as w
from opym.decon_config import decon_params_fingerprint
from opym.registry import StatusRegistry
from opym.stream.live import LIVE_LATEST_NAME, LIVE_STATUS_NAME

from backfill.live_index import LIVE_PREVIEW_NAME, LiveIndex

N_T, N_C, SHAPE = 4, 2, (4, 16, 20)


class Session:
    """What opym.stream.live leaves on disk for one streamed acquisition."""

    def __init__(self, tmp_path, base="Pak2_Stim", root_name="microscopy"):
        self.psf = tmp_path / "averaged_psf.tif"
        self.psf.write_bytes(b"psf")
        self.root = tmp_path / root_name
        self.leaf = self.root / "20261023_LCK-NG_JB" / base
        self.store = self.leaf / "viewer" / f"{base}_dsr.ome.zarr"
        self.dsr_dir = self.leaf / "decon_stage" / "Decon" / "DSR_decon"
        self.jobs = tmp_path / "jobs"
        for d in (self.dsr_dir, self.jobs):
            d.mkdir(parents=True)
        w.create_processed_store(self.store, n_t=N_T, n_c=N_C, shape_zyx=SHAPE, channel_labels=["mNG_488nm", "mSc_561nm"])
        self.done: list[int] = []
        self.sid = f"sid-{base}"
        (self.jobs / LIVE_LATEST_NAME).write_text(
            json.dumps(
                {
                    "session_id": self.sid,
                    "base_name": base,
                    "store": str(self.store),
                    "dsr_dir": str(self.dsr_dir),
                    "num_timepoints": N_T,
                    "n_channels": N_C,
                    "channel_labels": ["mNG_488nm", "mSc_561nm"],
                }
            )
        )
        self.status("running")

    @property
    def key(self):
        return str(self.leaf)

    def finish_timepoint(self, t):
        mip = w.image_group(self.store, w.MIP_SERIES, mode="r+")["0"]
        rng = np.random.default_rng(t)
        for c in range(N_C):
            mip[t, c, 0] = rng.integers(100, 4000, size=SHAPE[1:], dtype=np.uint16)
        self.done.append(t)
        w.write_progress(self.store, n_t=N_T, n_c=N_C, done=[[x, c] for x in self.done for c in range(N_C)], state="running")
        self.status("running")

    def status(self, state, *, psf=None, updated_at=None, failed=()):
        (self.dsr_dir / LIVE_STATUS_NAME).write_text(
            json.dumps(
                {
                    "state": state,
                    "session_id": self.sid,
                    "updated_at": time.time() if updated_at is None else updated_at,
                    "timepoints_done": list(self.done),
                    "timepoints_failed": list(failed),
                    "decon_psf": psf or str(self.psf),
                    "decon_params": decon_params_fingerprint(),
                }
            )
        )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    s = Session(tmp_path)
    monkeypatch.setenv("OPYM_DECON_PSF", str(s.psf))
    reg = StatusRegistry(tmp_path / "registry.sqlite3")
    index = LiveIndex(reg, [s.root], decon_psf=s.psf, mip_fps=12.0, jobs=s.jobs)
    yield s, reg, index
    reg.close()


def _movies(s):
    return s.leaf / "mip_movies"


def test_a_running_session_is_registered_at_once(setup):
    s, reg, index = setup
    index.tick()
    row = reg.get_dataset(s.key)
    assert row is not None
    # The dataset key and master file are what discovery gives it once the
    # raw stores drain: no second row appears later.
    assert row["master_file"] == str(s.leaf.parent / "Pak2_Stim_mNG_488nm.ome.zarr")
    # No stage row while it runs: the receiver's supersede guard and the batch
    # path are untouched.
    assert reg.get_stage(s.key, "deskew") is None


def test_the_preview_follows_each_timepoint(setup):
    s, reg, index = setup
    s.finish_timepoint(0)
    index.tick()
    preview = _movies(s) / LIVE_PREVIEW_NAME
    first = preview.read_bytes()
    assert reg.get_dataset(s.key)["actual_timepoints"] == 1
    assert reg.get_dataset(s.key)["expected_timepoints"] == N_T

    s.finish_timepoint(1)
    index.tick()
    assert preview.read_bytes() != first
    assert reg.get_dataset(s.key)["actual_timepoints"] == 2


def test_complete_makes_movies_without_any_raw_data(setup):
    s, reg, index = setup
    for t in range(N_T):
        s.finish_timepoint(t)
    index.tick()
    s.status("complete")
    index.tick()

    assert reg.is_stage_done(s.key, "deskew")
    assert reg.is_stage_done(s.key, "mip_encode")
    assert (_movies(s) / "Pak2_Stim_composite.webm").stat().st_size > 0
    assert not (_movies(s) / LIVE_PREVIEW_NAME).exists()  # the movie's poster takes over
    assert not any(s.leaf.parent.glob("*.ome.zarr"))  # nothing drained
    row = reg.get_dataset(s.key)
    assert (row["actual_timepoints"], row["expected_timepoints"]) == (N_T, N_T)
    assert not index.sessions  # finished sessions stop being followed


def test_an_aborted_session_counts_what_it_got(setup):
    s, reg, index = setup
    s.finish_timepoint(0)
    s.finish_timepoint(1)
    s.status("complete")
    index.tick()
    row = reg.get_dataset(s.key)
    assert reg.is_stage_done(s.key, "mip_encode")
    assert (row["actual_timepoints"], row["expected_timepoints"]) == (2, N_T)


@pytest.mark.parametrize(
    "status_kwargs",
    [
        {"state": "complete", "psf": "/psf/another.tif"},
        {"state": "complete", "failed": [3]},
        {"state": "failed"},
    ],
    ids=["other-psf", "failed-timepoints", "failed"],
)
def test_anything_else_is_left_to_the_batch_path(setup, status_kwargs):
    s, reg, index = setup
    s.finish_timepoint(0)
    s.status(**status_kwargs)
    index.tick()
    assert reg.get_stage(s.key, "deskew") is None
    assert reg.get_stage(s.key, "mip_encode") is None


def test_a_stale_heartbeat_is_not_shown_as_acquiring(setup):
    s, reg, index = setup
    s.status("running", updated_at=time.time() - 3600)
    index.tick()
    assert reg.get_dataset(s.key) is None
    assert not index.sessions


def test_a_session_outside_the_scanned_roots_is_ignored(tmp_path, monkeypatch):
    s = Session(tmp_path, root_name="elsewhere")
    with StatusRegistry(tmp_path / "registry.sqlite3") as reg:
        index = LiveIndex(reg, [tmp_path / "microscopy"], decon_psf=None, mip_fps=12.0, jobs=s.jobs)
        index.tick()
        assert reg.get_dataset(s.key) is None


def test_a_short_session_replaced_in_live_latest_still_finishes(setup, tmp_path):
    s, reg, index = setup
    s.finish_timepoint(0)
    index.tick()
    s.status("complete")
    # The next session starts before the next tick.
    nxt = json.loads((s.jobs / LIVE_LATEST_NAME).read_text())
    nxt["session_id"] = "sid-next"
    (s.jobs / LIVE_LATEST_NAME).write_text(json.dumps(nxt))
    index.tick()
    assert reg.is_stage_done(s.key, "mip_encode")


def test_a_finished_session_still_named_in_live_latest_is_not_picked_up_again(setup, capsys):
    s, reg, index = setup
    s.finish_timepoint(0)
    s.status("complete")
    index.tick()
    capsys.readouterr()
    index.tick()
    index.tick()
    assert not index.sessions
    assert capsys.readouterr().out == ""
