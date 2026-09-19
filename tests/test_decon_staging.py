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
