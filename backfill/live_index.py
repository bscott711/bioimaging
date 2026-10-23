"""Puts a streamed acquisition into the registry while it is being acquired,
so opym-dashboard shows it at once instead of after the session, the raw
drain and the next discovery pass (about 6.5 min after the session ended,
measured 2026-10-14, and nothing at all during it).

The live lane (opym.stream.live) already writes everything this needs, as
each timepoint finishes: `live_latest.json` in the jobs dir names the
session's processed store and DSR dir, `.live_status.json` there is its
heartbeat, and the store's Z-MIP series holds every finished timepoint. None
of it waits for the raw data to drain. Registry writes stay in the backfill
(see opym.stream.live), so this runs in the watch process, on its own thread,
every `LIVE_TICK_S` -- including while the live lease pauses the passes. It
reads two small JSON files and one 2-D slice per channel, and writes a JPEG
and a few registry rows; the movie is encoded once, when the session ends.

- Running: register the dataset under the key discovery will give it, keep
  its timepoint count current, and refresh `mip_movies/live_preview.jpg`
  from the newest finished timepoint. No stage row is written while the
  session runs: the receiver refuses to supersede a name with a `running`
  stage, and the batch path keeps owning anything that doesn't finish.
- Complete, with the provenance the backfill would use and every timepoint
  the lane reported in the store: deskew done (the store is the result, as
  in `pipeline._live_lane_owns_deskew`) and the MIP movies encoded from it.
- Failed, stale or mismatched: left alone. The next pass after the drain
  treats it exactly as it did before this module existed.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from opym import lanes
from opym.decon_config import decon_params_fingerprint
from opym.discovery import KIND_ZARR_PRECROPPED, LeafDataset
from opym.ome_zarr_writer import MIP_SERIES, complete_timepoints, image_group, read_progress
from opym.registry import StatusRegistry
from opym.stream.live import LIVE_LATEST_NAME, live_status_is_fresh, read_live_status
from opym.utils import mirror_root, resolve_output_base

from backfill.cli import MIP_ENCODE_LOCK, _run_mip_encode
from backfill.mip_movie import build_poster_from_frames

LIVE_TICK_S = 5.0
LIVE_PREVIEW_NAME = "live_preview.jpg"
# How long a session that never finished stays tracked (it stops being
# tracked sooner once finalized).
_FORGET_AFTER_S = 3600.0


@dataclass
class _Session:
    info: dict
    ds: LeafDataset
    store: Path
    dsr_dir: Path
    registered: bool = False
    counted: int | None = None
    preview_t: int | None = None
    finished: bool = False
    seen_at: float = 0.0


def _leaf_for_store(store: Path) -> Path | None:
    """Inverse of `opym.stream.live.live_viewer_store`: the store is
    `resolve_output_base(leaf)/viewer/<leaf name>_dsr.ome.zarr`."""
    base = store.parent.parent
    if store.parent.name != "viewer":
        return None
    try:
        leaf = Path("/") / base.relative_to(mirror_root())
    except ValueError:
        leaf = base
    return leaf if store.name == f"{leaf.name}_dsr.ome.zarr" else None


def dataset_for_session(info: dict, roots: list[Path]) -> LeafDataset | None:
    """The `LeafDataset` discovery will produce for this session once its raw
    stores have drained (`<raw_dir>/<base>_<label>.ome.zarr`, grouped under
    `<raw_dir>/<base>`), or None when it isn't under a scanned root -- then
    discovery would never see it either."""
    store = Path(info.get("store") or "")
    leaf = _leaf_for_store(store)
    if leaf is None:
        return None
    root = next((Path(r) for r in roots if leaf.is_relative_to(r)), None)
    if root is None:
        return None
    labels = info.get("channel_labels") or []
    raw = tuple(sorted(leaf.parent / f"{leaf.name}_{label}.ome.zarr" for label in labels))
    return LeafDataset(
        root=root,
        leaf_dir=leaf,
        # Planned paths: the raw stores may still be on /dev/shm. Discovery's
        # register_dataset refreshes master_file once they land.
        master_file=raw[0] if raw else store,
        kind=KIND_ZARR_PRECROPPED,
        channel_zarr_paths=raw,
    )


def _read_latest(jobs: Path) -> dict | None:
    try:
        info = json.loads((jobs / LIVE_LATEST_NAME).read_text())
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) and info.get("session_id") else None


class LiveIndex:
    """One `tick()` per `LIVE_TICK_S`; see the module docstring."""

    def __init__(
        self,
        registry: StatusRegistry,
        roots: list[Path],
        *,
        decon_psf: Path | None,
        mip_fps: float,
        jobs: Path | None = None,
        clock=time.time,
    ) -> None:
        self.registry = registry
        self.roots = [Path(r) for r in roots]
        self.decon_psf = decon_psf
        self.mip_fps = mip_fps
        self.jobs = jobs
        self.clock = clock
        self.sessions: dict[str, _Session] = {}
        # Sessions already handled: live_latest.json keeps naming the last one
        # until the next starts, and it must not be picked up again.
        self.finished: set[str] = set()

    def tick(self) -> None:
        now = self.clock()
        info = _read_latest(self.jobs or lanes.jobs_dir())
        if info and info["session_id"] not in self.sessions and info["session_id"] not in self.finished:
            ds = dataset_for_session(info, self.roots)
            if ds is not None:
                self.sessions[info["session_id"]] = _Session(
                    info=info, ds=ds, store=Path(info["store"]), dsr_dir=Path(info["dsr_dir"]), seen_at=now
                )
        # Every tracked session, not only the newest: a short one can end
        # after the next has already replaced live_latest.json.
        for sid, s in list(self.sessions.items()):
            try:
                self._step(s, now)
            except Exception as e:  # noqa: BLE001 - one session must not stop the others
                print(f"[live-index] {s.ds.dataset_key}: {e!r}")
            if s.finished or now - s.seen_at > _FORGET_AFTER_S:
                del self.sessions[sid]
                self.finished.add(sid)

    def _step(self, s: _Session, now: float) -> None:
        status = read_live_status(s.dsr_dir)
        if not status or status.get("session_id") != s.info["session_id"]:
            return
        state = status.get("state")
        if state == "running" and live_status_is_fresh(status, now):
            self._register(s)
            self._refresh(s)
        elif state == "complete":
            self._register(s)
            self._finalize(s, status)
            s.finished = True
        elif state == "failed" or not live_status_is_fresh(status, now):
            s.finished = True  # the batch path owns it from here, as before

    def _register(self, s: _Session) -> None:
        if s.registered:
            return
        ds = s.ds
        self.registry.register_dataset(
            ds.dataset_key, root=str(ds.root), leaf_dir=str(ds.leaf_dir), master_file=str(ds.master_file)
        )
        s.registered = True
        print(f"[live-index] {ds.dataset_key}: registered")

    def _refresh(self, s: _Session) -> None:
        done = complete_timepoints(read_progress(s.store))
        if len(done) != s.counted:
            self.registry.set_triage(
                s.ds.dataset_key,
                signal_flag="ok",
                expected_timepoints=s.info.get("num_timepoints"),
                actual_timepoints=len(done),
            )
            s.counted = len(done)
        if done and done[-1] != s.preview_t:
            arr = image_group(s.store, MIP_SERIES)["0"]
            t = done[-1]
            frames = {f"C{c}": arr[t, c, 0] for c in range(arr.shape[1])}
            build_poster_from_frames(frames, self._movies_dir(s), name=LIVE_PREVIEW_NAME)
            s.preview_t = t

    def _movies_dir(self, s: _Session) -> Path:
        return resolve_output_base(s.ds.leaf_dir) / "mip_movies"

    def _finalize(self, s: _Session, status: dict) -> None:
        key = s.ds.dataset_key
        psf = self.decon_psf
        if psf is None or status.get("decon_psf") != str(psf) or status.get("decon_params") != decon_params_fingerprint():
            print(f"[live-index] {key}: live provenance differs from the backfill's; left to the batch path")
            return
        if status.get("timepoints_failed"):
            print(f"[live-index] {key}: live lane failed some timepoints; left to the batch path")
            return
        have = complete_timepoints(read_progress(s.store))
        reported = sorted(int(t) for t in status.get("timepoints_done") or [])
        if not reported or not set(reported) <= set(have):
            print(f"[live-index] {key}: store has {len(have)} of {len(reported)} reported timepoints; left to the batch path")
            return
        with MIP_ENCODE_LOCK:
            if self.registry.is_stage_done(key, "mip_encode") and self.registry.is_stage_done(key, "deskew"):
                return
            self.registry.set_decon_psf(key, str(psf))
            self.registry.set_decon_params(key, decon_params_fingerprint())
            self.registry.finish_stage(key, "deskew", status="done", output_path=str(s.store))
            start = time.monotonic()
            _run_mip_encode(s.ds, self.registry, self.mip_fps, declared_timepoints=s.info.get("num_timepoints"))
        if self.registry.is_stage_done(key, "mip_encode"):
            (self._movies_dir(s) / LIVE_PREVIEW_NAME).unlink(missing_ok=True)
            print(f"[live-index] {key}: session complete, movies in {time.monotonic() - start:.1f}s")


def start_live_index_thread(
    registry_path: Path, roots: list[Path], *, decon_psf: Path | None, mip_fps: float
) -> threading.Thread:
    """Runs `LiveIndex.tick` every `LIVE_TICK_S` on a daemon thread with its
    own registry connection (sqlite connections aren't shared across threads)."""

    def loop() -> None:
        index = LiveIndex(StatusRegistry(registry_path), roots, decon_psf=decon_psf, mip_fps=mip_fps)
        while True:
            try:
                index.tick()
            except Exception as e:  # noqa: BLE001 - the thread must outlive one bad tick
                print(f"[live-index] tick failed: {e!r}")
            time.sleep(LIVE_TICK_S)

    thread = threading.Thread(target=loop, name="live-index", daemon=True)
    thread.start()
    return thread
