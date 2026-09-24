"""Contract for `build_decon_staging_dir` -- the zarr-path decon input.

Deconvolution cannot read the symlink mirror: PetaKit5D's decon path has no
axis-order parameter and convolves the array exactly as stored, while the
mirror presents (z, y, x). This step rewrites the pixels into the
(ny, nx, nz) layout decon requires, which is also the layout the legacy
OME-TIFF path has always used and the layout the measured PSF matches.

Pure Python -- builds tiny fake OME-NGFF stores in tmp_path, no MATLAB, no
GPU, no real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import zarr

from opym.discovery import KIND_ZARR_PRECROPPED
from backfill.pipeline import (
    build_decon_staging_dir,
    build_zarr_pyramid_mirror,
    dsr_dir_name_for,
    dsr_output_dir,
    resolve_decon_psf,
)
from opym.utils import orient_zyx_for_decon_tiff


def _make_4d_store(path: Path, data: np.ndarray) -> Path:
    """Minimal stand-in for a pymmcore MDA store: pixels under `p0/`, with
    the multiscales metadata `_read_ome_zarr_dataset_path` reads."""
    root = zarr.open(str(path), mode="w")
    # `dimension_separator="/"` matches what the real pymmcore writer emits
    # (verified against a live store's .zarray), which is what makes each
    # timepoint a directory `p0/<t>/` -- the thing both this staging step and
    # `build_zarr_pyramid_mirror` use to tell a WRITTEN timepoint from one the
    # store merely declares. With `fill_value: 0`, an unwritten timepoint
    # would otherwise read back as silent zeros.
    arr = root.create_dataset(
        "p0",
        shape=data.shape,
        chunks=(1, 1) + data.shape[2:],
        dtype=data.dtype,
        dimension_separator="/",
    )
    arr[:] = data
    root.attrs["multiscales"] = [
        {
            "axes": [{"name": n} for n in "tzyx"],
            "datasets": [
                {"path": "p0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 1]}]}
            ],
        }
    ]
    return path


@pytest.fixture
def two_channel_store(tmp_path):
    """2 channels x 3 timepoints, with 3 distinct spatial extents so a wrong
    permutation cannot coincidentally pass."""
    nt, nz, ny_tilted, nx_coverslip = 3, 7, 5, 11
    stores, volumes = [], {}
    rng = np.random.default_rng(0)
    for c in range(2):
        data = rng.integers(0, 4000, size=(nt, nz, ny_tilted, nx_coverslip), dtype=np.uint16)
        p = _make_4d_store(tmp_path / f"cell_001_ch{c}.ome.zarr", data)
        stores.append(p)
        volumes[c] = data
    return tuple(stores), volumes, tmp_path


def test_staged_frames_are_oriented_for_decon(two_channel_store):
    stores, volumes, tmp_path = two_channel_store
    out = build_decon_staging_dir(stores, tmp_path / "decon_stage", dataset_prefix="cell_001")

    staged = tifffile.imread(out / "cell_001_C0_T000.tif")
    assert np.array_equal(staged, orient_zyx_for_decon_tiff(volumes[0][0]))
    # (nz, ny, nx) on disk -> MATLAB readtiff gives (ny, nx, nz)
    assert staged.shape == (7, 11, 5)


def test_staged_names_match_the_channel_patterns_the_ticket_declares(two_channel_store):
    """`submit_zarr_deskew_ticket` declares `_C{i}_T` patterns for a time
    series; PetaKit5D matches them by substring against these filenames."""
    stores, _, tmp_path = two_channel_store
    out = build_decon_staging_dir(stores, tmp_path / "decon_stage", dataset_prefix="cell_001")

    names = sorted(p.name for p in out.glob("*.tif"))
    assert names == [f"cell_001_C{c}_T{t:03d}.tif" for c in (0, 1) for t in range(3)]
    for c in (0, 1):
        assert sum(f"_C{c}_T" in n for n in names) == 3


def test_max_timepoints_caps_the_stage(two_channel_store):
    """OPYM_ZARR_MAX_TIMEPOINTS=1 must give a one-timepoint run, the fast
    validation loop."""
    stores, _, tmp_path = two_channel_store
    out = build_decon_staging_dir(
        stores, tmp_path / "decon_stage", dataset_prefix="cell_001", max_timepoints=1
    )
    assert sorted(p.name for p in out.glob("*.tif")) == [
        "cell_001_C0_T000.tif",
        "cell_001_C1_T000.tif",
    ]


def test_staging_is_idempotent_and_prunes_stale_frames(two_channel_store):
    """A rebuild clamped to fewer timepoints must remove the extra frames.

    Leaving them behind is the bug class that broke Cell_002's mip_encode:
    a stale frame still matches `_C{c}_T`, gets processed, and the
    per-channel counts diverge.
    """
    stores, _, tmp_path = two_channel_store
    staging = tmp_path / "decon_stage"

    build_decon_staging_dir(stores, staging, dataset_prefix="cell_001")
    assert len(list(staging.glob("*.tif"))) == 6
    mtimes = {p.name: p.stat().st_mtime_ns for p in staging.glob("*.tif")}

    build_decon_staging_dir(stores, staging, dataset_prefix="cell_001")
    assert {p.name: p.stat().st_mtime_ns for p in staging.glob("*.tif")} == mtimes, (
        "an unchanged frame must not be rewritten"
    )

    build_decon_staging_dir(stores, staging, dataset_prefix="cell_001", max_timepoints=1)
    assert len(list(staging.glob("*.tif"))) == 2
    assert not (staging / "cell_001_C0_T002.tif").exists()


def test_declared_but_unwritten_timepoints_are_skipped(tmp_path):
    """An aborted acquisition declares more timepoints than it wrote.

    Real example: stores declaring 100 while holding 29-80, with the two
    channels disagreeing. Because `fill_value` is 0, staging such a timepoint
    would silently produce an all-zero frame that still matches `_C{c}_T` and
    gets deconvolved and deskewed like real data.
    """
    data = np.ones((4, 7, 5, 11), dtype=np.uint16)
    store = _make_4d_store(tmp_path / "aborted.ome.zarr", data)
    # Simulate the abort: drop the chunk directories for the last two
    # timepoints while leaving the declared shape at 4.
    import shutil

    for t in (2, 3):
        shutil.rmtree(store / "p0" / str(t))

    out = build_decon_staging_dir((store,), tmp_path / "decon_stage", dataset_prefix="ab")
    assert sorted(p.name for p in out.glob("*.tif")) == ["ab_C0_T000.tif", "ab_C0_T001.tif"]


def test_only_one_real_timepoint_uses_single_timepoint_naming(tmp_path):
    """Regression for a real failure (`.../20260710-YG_PSF/bead_004`): a
    store DECLARES more than one timepoint but only ever wrote one real
    chunk (an acquisition aborted after its very first timepoint). The
    naming decision here must agree with `dataset_timepoints()` (which
    counts real chunks the same way and returns 1 for this store), or the
    caller submits a deskew ticket with the single-timepoint
    `channel_patterns` (`<name>.ome`) while this function actually staged
    `<prefix>_C0_T000.tif` -- a pattern mismatch that made PetaKit5D's
    `getImageSize('')` die with "Index exceeds array bounds" on the real
    dataset. `test_max_timepoints_caps_the_stage` above is the opposite
    case (genuinely multi-timepoint, just capped for a fast test run) and
    must keep multi-style naming -- this asserts the two are told apart by
    the real written-chunk count, not by whichever count happens to be
    small.
    """
    data = np.ones((2, 7, 5, 11), dtype=np.uint16)
    store = _make_4d_store(tmp_path / "bead_004_GFP_488.ome.zarr", data)
    import shutil

    shutil.rmtree(store / "p0" / "1")  # only T000 was ever really written

    out = build_decon_staging_dir((store,), tmp_path / "decon_stage", dataset_prefix="bead_004")
    assert [p.name for p in out.glob("*.tif")] == ["bead_004_GFP_488.ome.tif"]


def test_zarr_mirror_only_one_real_timepoint_uses_single_timepoint_naming(tmp_path):
    """`build_zarr_pyramid_mirror` (the deskew-only, non-decon counterpart)
    has the identical naming decision and the identical bug potential --
    same fix, same reasoning as
    `test_only_one_real_timepoint_uses_single_timepoint_naming` above.
    """
    data = np.ones((2, 7, 5, 11), dtype=np.uint16)
    store = _make_4d_store(tmp_path / "bead_004_GFP_488.ome.zarr", data)
    import shutil

    shutil.rmtree(store / "p0" / "1")

    out = build_zarr_pyramid_mirror((store,), tmp_path / "zarr_mirror", dataset_prefix="bead_004")
    assert [p.name for p in out.iterdir()] == ["bead_004_GFP_488.ome.zarr"]


def test_single_timepoint_keeps_the_ome_bearing_store_name(tmp_path):
    """`_run_mip_encode`'s poster branch matches PetaKit5D's MIP output
    against the store name minus only `.zarr` -- so the `.ome` component
    must survive into the staged filename."""
    data = np.zeros((1, 7, 5, 11), dtype=np.uint16)
    store = _make_4d_store(tmp_path / "cell_003_GFP_488.ome.zarr", data)
    out = build_decon_staging_dir((store,), tmp_path / "decon_stage")
    assert [p.name for p in out.glob("*.tif")] == ["cell_003_GFP_488.ome.tif"]


def test_no_tmp_files_survive(two_channel_store):
    """Frames are written to `.tmp` then renamed -- a surviving partial file
    would be handed straight back by the skip-if-present check and poison
    every retry."""
    stores, _, tmp_path = two_channel_store
    out = build_decon_staging_dir(stores, tmp_path / "decon_stage", dataset_prefix="cell_001")
    assert list(out.glob("*.tmp")) == []


# --------------------------------------------------------------------------
# Decon switch / output naming
# --------------------------------------------------------------------------


def test_decon_is_off_by_default(monkeypatch):
    monkeypatch.delenv("OPYM_DECON_PSF", raising=False)
    assert resolve_decon_psf() is None
    assert dsr_dir_name_for(None) == "DSR_nodecon"


def test_decon_psf_enables_a_separate_output_dir(monkeypatch, tmp_path):
    psf = tmp_path / "psf.tif"
    psf.touch()
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    assert resolve_decon_psf() == psf.resolve()
    assert dsr_dir_name_for(psf) == "DSR_decon"
    assert dsr_dir_name_for(psf) != dsr_dir_name_for(None), (
        "deconvolved output must never overwrite the no-decon archive"
    )


def test_a_missing_psf_raises_rather_than_silently_skipping_decon(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYM_DECON_PSF", str(tmp_path / "absent.tif"))
    with pytest.raises(FileNotFoundError):
        resolve_decon_psf()


def test_decon_dsr_output_nests_under_decon(tmp_path):
    """PetaKit5D writes DSR inside whatever it gets as `dataDir`, and when
    decon runs the deskew step is handed `<dataDir>/Decon` instead -- so the
    DSR result is one level deeper. Confirmed against a real completed job;
    reading the shallow path is the "No MIP TIFFs found" failure all over again.
    """
    data_dir = tmp_path / "decon_stage"
    psf = tmp_path / "psf.tif"
    psf.touch()
    assert dsr_output_dir(data_dir, None) == data_dir / "DSR_nodecon"
    assert dsr_output_dir(data_dir, psf) == data_dir / "Decon" / "DSR_decon"


def test_reaper_keeps_the_dsr_output(tmp_path, monkeypatch):
    """`_reap_decon_intermediates` used to rmtree `Decon/` wholesale -- but
    PetaKit5D nests the final DSR result INSIDE it, at `Decon/DSR_decon`
    (run_petakit_server.m sets current_input_dir = <dataDir>/Decon before the
    deskew step). So the cleanup deleted the one output it exists to keep,
    on every decon-enabled dataset.
    """
    from backfill import cli

    leaf = tmp_path / "Cell_001"
    stage = leaf / "decon_stage"
    decon = stage / "Decon"
    dsr = decon / "DSR_decon"
    dsr.mkdir(parents=True)
    (decon / "psfgen").mkdir()
    (decon / "psfgen" / "psf_omw.tif").write_bytes(b"psf")
    (decon / "Cell_001_C0_T000.tif").write_bytes(b"decon frame")   # intermediate
    (dsr / "Cell_001_C0_T000.tif").write_bytes(b"final DSR")       # the output
    (stage / "Cell_001_C0_T000.tif").write_bytes(b"staged input")  # intermediate

    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"x")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.delenv("OPYM_KEEP_DECON_INTERMEDIATES", raising=False)

    ds = SimpleNamespace(
        kind=KIND_ZARR_PRECROPPED,
        leaf_dir=leaf,
        dataset_key="k",
        master_file=leaf / "raw.ome.tif",
        channel_zarr_paths=[],
    )
    cli._reap_decon_intermediates(ds)

    assert (dsr / "Cell_001_C0_T000.tif").exists(), "final DSR output was deleted"
    assert not (decon / "Cell_001_C0_T000.tif").exists(), "decon intermediate kept"
    assert not (stage / "Cell_001_C0_T000.tif").exists(), "staged input kept"
    assert (leaf / "decon_qc" / "psf_omw.tif").exists(), "psfgen QC not preserved"


# --------------------------------------------------------------------------
# Re-run safety: stale output must not survive a configuration change
# --------------------------------------------------------------------------


class _FakeRegistry:
    """Only the four accessors the provenance/submit path actually touches."""

    def __init__(self, rows=None):
        self.rows = rows or {}
        # The real `all_datasets()` selects * from datasets, so every row it
        # returns carries its own key. Mirror that rather than making each
        # test repeat it.
        for key, row in self.rows.items():
            row.setdefault("dataset_key", key)
        self.started = []

    def _row(self, key):
        return self.rows.setdefault(key, {"dataset_key": key})

    def get_stage(self, key, stage):
        return self._row(key).get(f"stage:{stage}")

    def is_stage_done(self, key, stage):
        row = self.get_stage(key, stage)
        return bool(row and row["status"] == "done")

    def get_decon_psf(self, key):
        return self._row(key).get("decon_psf")

    def get_decon_params(self, key):
        return self._row(key).get("decon_params")

    def set_decon_psf(self, key, value):
        self._row(key)["decon_psf"] = value

    def set_decon_params(self, key, value):
        self._row(key)["decon_params"] = value

    def register_dataset(self, key, **kw):
        self._row(key).update(kw)

    def start_stage(self, key, stage, *, ticket_path=None):
        self._row(key)[f"stage:{stage}"] = {"status": "running", "ticket_path": ticket_path}
        self.started.append((key, stage))

    def all_datasets(self):
        return list(self.rows.values())


def test_a_parameter_retune_clears_decon_output_from_the_previous_settings(
    two_channel_store, monkeypatch, capsys
):
    """Regression for the 2026-09-21 failure.

    Locking in the `super4` OMW parameters made `decon_provenance_matches`
    (correctly) reject 13 already-`done` datasets, so they re-submitted. But
    the stale-output cleanup was gated on `status == "failed"`, and these rows
    were `done` -- so the previous run's `Decon/` stayed on disk. Two separate
    ways that bites, both of which this asserts against:

    * `Decon/Masks/<fsname>_eroded.zarr` survives the intermediate reaper, and
      its existence sends XR_RLdeconFrame3D.m:244 into `rmdirs`, which is not a
      function anywhere in the vendored PetaKit5D. All 13 died there.
    * Even with that shimmed, PetaKit5D skips any decon frame whose output
      already exists, so the old-alpha pixels would have been reused and then
      stamped with the NEW provenance fingerprint -- silently wrong data
      reporting itself correct.
    """
    from backfill import pipeline

    stores, _volumes, tmp_path = two_channel_store
    leaf = tmp_path
    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.setenv("OPYM_ZARR_DEFAULT_Z_STEP", "0.5")
    monkeypatch.delenv("OPYM_DECON_REPROCESS_LEGACY", raising=False)

    submitted = {}

    def _fake_submit(**kw):
        submitted.update(kw)
        ticket = tmp_path / "ticket.json"
        ticket.write_text("{}")
        return ticket

    monkeypatch.setattr(pipeline, "submit_remote_deskew_job", _fake_submit)

    # Output from the PREVIOUS run, at the old parameters.
    data_dir = tmp_path / "decon_stage"
    decon = data_dir / "Decon"
    (decon / "Masks").mkdir(parents=True)
    (decon / "Masks" / "cell_001_C0_T000_eroded.zarr").mkdir()
    (decon / "DSR_decon").mkdir()
    (decon / "DSR_decon" / "cell_001_C0_T000.tif").write_bytes(b"old alpha=0.02 output")
    (decon / "cell_001_C0_T000.tif").write_bytes(b"old decon intermediate")

    ds = SimpleNamespace(
        kind=KIND_ZARR_PRECROPPED,
        leaf_dir=leaf,
        raw_dir=leaf,
        dataset_key="k",
        master_file=leaf / "raw.ome.tif",
        channel_zarr_paths=tuple(stores),
    )
    registry = _FakeRegistry(
        {
            "k": {
                "dataset_key": "k",
                "stage:deskew": {"status": "done", "ticket_path": "old.json"},
                "decon_psf": str(psf.resolve()),
                "decon_params": "a0.02_o0.9_h0.8-1.0_d1",  # the OLD settings
            }
        }
    )

    ticket = pipeline.submit_zarr_deskew_ticket(ds, registry)

    assert ticket is not None, "a parameter change must re-submit, not skip"
    assert not decon.exists(), (
        "Decon/ from the previous parameter set survived -- the eroded mask "
        "crashes the re-run in rmdirs, and any surviving decon frame is "
        "silently reused instead of recomputed"
    )
    assert submitted["wiener_alpha"] == pipeline.DECON_WIENER_ALPHA
    assert registry.get_decon_params("k") == pipeline.decon_params_fingerprint()


def test_legacy_decon_output_is_grandfathered_unless_opted_in(monkeypatch, tmp_path):
    """A NULL `decon_params` means "deconvolved before the fingerprint column
    existed", not "made with the wrong settings we can prove".

    Reading it as stale re-submits the entire legacy corpus (65 datasets as of
    2026-09-21) the instant the backfill restarts. Held by default; opt in with
    OPYM_DECON_REPROCESS_LEGACY=1.
    """
    from backfill import pipeline

    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    registry = _FakeRegistry(
        {
            "legacy": {"decon_psf": str(psf), "decon_params": None},
            "current": {"decon_psf": str(psf), "decon_params": pipeline.decon_params_fingerprint()},
            "retuned": {"decon_psf": str(psf), "decon_params": "a0.02_o0.9_h0.8-1.0_d1"},
            "other-psf": {"decon_psf": "/some/other.tif", "decon_params": None},
        }
    )

    monkeypatch.delenv("OPYM_DECON_REPROCESS_LEGACY", raising=False)
    assert pipeline.decon_provenance_matches(registry, "legacy", psf) is True
    assert pipeline.decon_provenance_matches(registry, "current", psf) is True
    assert pipeline.decon_provenance_matches(registry, "retuned", psf) is False, (
        "a known, differing fingerprint is a real mismatch and must still re-run"
    )
    assert pipeline.decon_provenance_matches(registry, "other-psf", psf) is False, (
        "grandfathering must not reach across a PSF change"
    )

    monkeypatch.setenv("OPYM_DECON_REPROCESS_LEGACY", "1")
    assert pipeline.decon_provenance_matches(registry, "legacy", psf) is False
    assert pipeline.decon_provenance_matches(registry, "current", psf) is True


def test_grandfathered_datasets_are_reported_not_silent(monkeypatch, tmp_path, capsys):
    """The whole point of the fingerprint was to stop a parameter change
    looking like a no-op. Holding a cohort back reintroduces exactly that, so
    it has to be said out loud once per pass.
    """
    from backfill import pipeline

    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    done = {"status": "done", "ticket_path": "t.json"}
    registry = _FakeRegistry(
        {
            "a": {"decon_psf": str(psf), "decon_params": None, "stage:deskew": done},
            "b": {"decon_psf": str(psf), "decon_params": None, "stage:deskew": done},
            "c": {
                "decon_psf": str(psf),
                "decon_params": pipeline.decon_params_fingerprint(),
                "stage:deskew": done,
            },
            "d": {"decon_psf": None, "decon_params": None},  # never deconvolved
            # NULL fingerprint but no finished deskew: nothing is being held
            # back here, so counting it overstates the exemption (539 vs 65 on
            # the real registry).
            "e": {
                "decon_psf": str(psf),
                "decon_params": None,
                "stage:deskew": {"status": "failed", "ticket_path": "t.json"},
            },
            "f": {"decon_psf": str(psf), "decon_params": None},  # no deskew row at all
        }
    )

    monkeypatch.delenv("OPYM_DECON_REPROCESS_LEGACY", raising=False)
    assert pipeline.log_grandfathered_decon_datasets(registry, psf) == 2
    assert "2 dataset(s)" in capsys.readouterr().out

    monkeypatch.setenv("OPYM_DECON_REPROCESS_LEGACY", "1")
    assert pipeline.log_grandfathered_decon_datasets(registry, psf) == 0
    assert pipeline.log_grandfathered_decon_datasets(registry, None) == 0


def _tiff_dataset(tmp_path):
    """A legacy OME-TIFF dataset whose crop stage already ran, so
    `resolve_deskew_working_dir` resolves to the master-stem directory."""
    leaf = tmp_path / "Cell_1"
    leaf.mkdir()
    master = leaf / "cell_MMStack_Pos0.ome.tif"
    master.write_bytes(b"raw")
    work = leaf / "cell_MMStack_Pos0"
    work.mkdir()
    (work / "cell_MMStack_Pos0_C0_T000.tif").write_bytes(b"cropped frame")
    return (
        SimpleNamespace(
            kind="tiff",
            leaf_dir=leaf,
            raw_dir=leaf,
            dataset_key="k",
            master_file=master,
            channel_zarr_paths=(),
        ),
        work,
    )


def test_tiff_path_does_not_delete_decon_output_it_never_created(tmp_path, monkeypatch):
    """The crop stage's working dir is also where the older manual `opym` CLI
    wrote its own `Decon/`. With no registry row there is nothing to say that
    output came from this pipeline, so it is not ours to remove -- the
    stale-output cleanup must stay out of a dataset the backfill has never
    submitted for.
    """
    from backfill import pipeline

    ds, work = _tiff_dataset(tmp_path)
    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.setattr(
        pipeline, "submit_remote_deskew_job", lambda **kw: tmp_path / "ticket.json"
    )
    (tmp_path / "ticket.json").write_text("{}")

    hand_run = work / "Decon"
    hand_run.mkdir()
    (hand_run / "precious.tif").write_bytes(b"hand-run result")

    registry = _FakeRegistry({"k": {"dataset_key": "k"}})  # no deskew row at all
    pipeline.submit_deskew_ticket(ds, work, registry)

    assert (hand_run / "precious.tif").exists(), (
        "deleted Decon/ output that this pipeline never produced"
    )


def test_tiff_path_clears_its_own_stale_output_on_a_retune(tmp_path, monkeypatch):
    """Counterpart to the test above: once a registry row exists, the output
    IS this pipeline's, and a provenance mismatch must clear it -- the same
    `done`-row retune case that broke the zarr path on 2026-09-21.
    """
    from backfill import pipeline

    ds, work = _tiff_dataset(tmp_path)
    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.setattr(
        pipeline, "submit_remote_deskew_job", lambda **kw: tmp_path / "ticket.json"
    )
    (tmp_path / "ticket.json").write_text("{}")

    stale = work / "Decon"
    (stale / "Masks").mkdir(parents=True)
    (stale / "Masks" / "cell_MMStack_Pos0_C0_T000_eroded.zarr").mkdir()
    (stale / "old.tif").write_bytes(b"old alpha")

    registry = _FakeRegistry(
        {
            "k": {
                "dataset_key": "k",
                "stage:deskew": {"status": "done", "ticket_path": "old.json"},
                "decon_psf": str(psf.resolve()),
                "decon_params": "a0.02_o0.9_h0.8-1.0_d1",
            }
        }
    )
    pipeline.submit_deskew_ticket(ds, work, registry)

    assert not stale.exists(), "stale Decon/ survived a parameter retune"


# --- priority lanes: backfill admission (opym.lanes) ----------------------


def _resubmission_scenario(two_channel_store, monkeypatch):
    """A `done` dataset whose decon params changed, so the next pass would
    clean its old output and submit (see the test above)."""
    from backfill import pipeline

    stores, _volumes, tmp_path = two_channel_store
    psf = tmp_path / "psf.tif"
    psf.write_bytes(b"psf")
    monkeypatch.setenv("OPYM_DECON_PSF", str(psf))
    monkeypatch.setenv("OPYM_ZARR_DEFAULT_Z_STEP", "0.5")
    monkeypatch.delenv("OPYM_DECON_REPROCESS_LEGACY", raising=False)
    submitted = []

    def _fake_submit(**kw):
        submitted.append(kw)
        ticket = tmp_path / "ticket.json"
        ticket.write_text("{}")
        return ticket

    monkeypatch.setattr(pipeline, "submit_remote_deskew_job", _fake_submit)
    old_output = tmp_path / "decon_stage" / "Decon" / "DSR_decon" / "cell_001_C0_T000.tif"
    old_output.parent.mkdir(parents=True)
    old_output.write_bytes(b"old output")
    ds = SimpleNamespace(
        kind=KIND_ZARR_PRECROPPED,
        leaf_dir=tmp_path,
        raw_dir=tmp_path,
        dataset_key="k",
        master_file=tmp_path / "raw.ome.tif",
        channel_zarr_paths=tuple(stores),
    )
    registry = _FakeRegistry(
        {
            "k": {
                "dataset_key": "k",
                "stage:deskew": {"status": "done", "ticket_path": "old.json"},
                "decon_psf": str(psf.resolve()),
                "decon_params": "a0.02_o0.9_h0.8-1.0_d1",
            }
        }
    )
    return pipeline, ds, registry, submitted, old_output


def test_zarr_submit_is_deferred_during_a_live_lease_before_any_cleanup(
    two_channel_store, monkeypatch
):
    from opym import lanes

    pipeline, ds, registry, submitted, old_output = _resubmission_scenario(
        two_channel_store, monkeypatch
    )
    lanes.write_lease(["live-session"])

    assert pipeline.submit_zarr_deskew_ticket(ds, registry) is None
    assert submitted == []
    # Deferral happens before the stale-output cleanup, so a dataset waiting out
    # a live acquisition doesn't lose its old output with nothing queued.
    assert old_output.exists()
    assert registry.get_stage("k", "deskew")["status"] == "done"


def test_zarr_submit_is_deferred_at_the_inflight_cap(two_channel_store, monkeypatch):
    from opym import lanes

    pipeline, ds, registry, submitted, _ = _resubmission_scenario(
        two_channel_store, monkeypatch
    )
    monkeypatch.setenv("OPYM_BACKFILL_MAX_INFLIGHT", "1")
    lanes.backfill_queue_dir().mkdir(parents=True)
    (lanes.backfill_queue_dir() / ".active_other.json").write_text("{}")

    assert pipeline.submit_zarr_deskew_ticket(ds, registry) is None
    assert submitted == []


def test_zarr_submit_proceeds_under_the_inflight_cap(two_channel_store, monkeypatch):
    from opym import lanes

    pipeline, ds, registry, submitted, _ = _resubmission_scenario(
        two_channel_store, monkeypatch
    )
    monkeypatch.setenv("OPYM_BACKFILL_MAX_INFLIGHT", "2")
    lanes.backfill_queue_dir().mkdir(parents=True)
    (lanes.backfill_queue_dir() / "other.json").write_text("{}")

    assert pipeline.submit_zarr_deskew_ticket(ds, registry) is not None
    assert len(submitted) == 1


def test_backfill_tickets_use_the_shared_decon_config(two_channel_store, monkeypatch):
    """Both lanes build tickets from opym.decon_config.deskew_decon_kwargs, so
    live and batch output can't drift apart. Pins the 2026-09-24 switch to
    linear DSR interpolation (cubic took 13.6 s/frame on CPU) and that the
    super4 decon parameters are unchanged by the move."""
    pipeline, ds, registry, submitted, _ = _resubmission_scenario(
        two_channel_store, monkeypatch
    )
    assert pipeline.submit_zarr_deskew_ticket(ds, registry) is not None
    kw = submitted[0]
    assert kw["interp_method"] == "linear"
    assert kw["xy_pixel_size"] == 0.136
    assert kw["dsr_dir_name"] == "DSR_decon"
    assert (kw["wiener_alpha"], kw["otf_cum_thresh"], kw["hann_win_bounds"]) == (0.20, 0.90, [0.4, 1.0])
    assert (kw["damp_factor"], kw["edge_erosion"], kw["gpu_decon"]) == (2, 3, True)
    assert pipeline.decon_params_fingerprint() == "a0.2_o0.9_h0.4-1.0_d2"
