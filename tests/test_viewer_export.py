"""Viewer-export invariants.

The point of this module is that DSR output opens in ChimeraX and napari with
real units and no extra steps. The viewer behaviour itself was verified by
opening real DSR frames in both (ChimeraX returns step=(0.136,0.136,0.136)
from an OME-TIFF; napari returns scale (1,0.136,0.136,0.136) from the
OME-Zarr and ignores TIFF metadata entirely). These tests pin the file-level
facts that make that work, so a refactor can't quietly undo them.
"""

import json

import numpy as np
import pytest
import tifffile

from backfill.viewer_export import (
    DSR_VOXEL_UM,
    channel_label,
    export_for_viewers,
    parse_dsr_frames,
    stamp_ome_tiff,
    write_ome_zarr,
)


def _make_frames(d, *, channels=2, times=3, shape=(8, 12, 10), compression="lzw"):
    """A miniature DSR output directory, LZW like PetaKit5D's own writetiff."""
    rng = np.random.default_rng(0)
    for c in range(channels):
        for t in range(times):
            vol = (rng.random(shape) * 1000).astype(np.uint16)
            tifffile.imwrite(d / f"Cell_001_C{c}_T{t:03d}.tif", vol, compression=compression)
    return d


def test_parse_frames_ignores_mips(tmp_path):
    _make_frames(tmp_path, channels=2, times=2)
    # PetaKit5D drops MIPs beside the frames; they must not be mistaken for data.
    tifffile.imwrite(tmp_path / "Cell_001_C0_T000_MIP_z.tif", np.zeros((4, 4), np.uint16))
    frames = parse_dsr_frames(tmp_path)
    assert set(frames) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_channel_label():
    assert channel_label("Cell_002_GFP_488.ome.zarr") == "GFP 488"
    assert channel_label("Cell_002_mScarlet_561.ome.zarr") == "mScarlet 561"
    # No <fluor>_<wavelength> tail: degrade to something readable, don't raise.
    assert channel_label("weird_name.zarr") == "weird_name"


def test_stamp_writes_voxel_size_and_keeps_pixels(tmp_path):
    _make_frames(tmp_path, channels=1, times=1)
    f = tmp_path / "Cell_001_C0_T000.tif"
    before = tifffile.imread(f)

    assert stamp_ome_tiff(f, channel="GFP 488") is True
    after = tifffile.imread(f)
    np.testing.assert_array_equal(before, after)

    with tifffile.TiffFile(f) as tf:
        assert tf.is_ome
        xml = tf.ome_metadata
    for axis in ("X", "Y", "Z"):
        assert f'PhysicalSize{axis}="{DSR_VOXEL_UM}"' in xml


def test_stamp_preserves_compression(tmp_path):
    """Rewriting uncompressed took one real 85 MB frame pair to 914 MB; across
    the ~1800 frames of this corpus that is tens of terabytes."""
    _make_frames(tmp_path, channels=1, times=1, compression="lzw")
    f = tmp_path / "Cell_001_C0_T000.tif"
    with tifffile.TiffFile(f) as tf:
        before = tf.pages[0].compression
    stamp_ome_tiff(f)
    with tifffile.TiffFile(f) as tf:
        assert tf.pages[0].compression == before


def test_stamp_is_idempotent(tmp_path):
    _make_frames(tmp_path, channels=1, times=1)
    f = tmp_path / "Cell_001_C0_T000.tif"
    assert stamp_ome_tiff(f) is True
    assert stamp_ome_tiff(f) is False       # already OME: no second rewrite


def test_ome_zarr_axes_scale_and_pyramid(tmp_path):
    zarr = pytest.importorskip("zarr")
    src = tmp_path / "dsr"
    src.mkdir()
    _make_frames(src, channels=2, times=3, shape=(8, 12, 10))
    out = write_ome_zarr(src, tmp_path / "x.ome.zarr", channel_labels=["GFP 488", "mScarlet 561"])

    g = zarr.open_group(str(out), mode="r")
    ms = g.attrs["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["t", "c", "z", "y", "x"]
    # Spatial scale is the DSR voxel size, doubling per pyramid level; t and c
    # are not spatial and must stay 1 or napari mis-scales the time slider.
    for lvl, dset in enumerate(ms["datasets"]):
        scale = dset["coordinateTransformations"][0]["scale"]
        assert scale[1] == 1.0
        assert scale[2:] == [DSR_VOXEL_UM * 2 ** lvl] * 3
    assert g["0"].shape == (3, 2, 8, 12, 10)
    assert g["1"].shape[2:] == (4, 6, 5)
    assert [c["label"] for c in g.attrs["omero"]["channels"]] == ["GFP 488", "mScarlet 561"]


def test_ome_zarr_round_trips_pixels(tmp_path):
    zarr = pytest.importorskip("zarr")
    src = tmp_path / "dsr"
    src.mkdir()
    _make_frames(src, channels=1, times=2, shape=(6, 8, 8))
    out = write_ome_zarr(src, tmp_path / "y.ome.zarr", levels=1)
    g = zarr.open_group(str(out), mode="r")
    for t in range(2):
        expected = tifffile.imread(src / f"Cell_001_C0_T{t:03d}.tif")
        np.testing.assert_array_equal(g["0"][t, 0], expected)


def test_ome_zarr_needs_frames(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_ome_zarr(tmp_path, tmp_path / "empty.ome.zarr")


def test_export_writes_everything(tmp_path):
    pytest.importorskip("zarr")
    src = tmp_path / "dsr"
    src.mkdir()
    _make_frames(src, channels=2, times=2, shape=(6, 8, 8))
    out = tmp_path / "viewer"
    summary = export_for_viewers(src, out, name="Cell_001",
                                 channel_labels=["GFP 488", "mScarlet 561"])

    assert summary["frames"] == 4 and summary["stamped"] == 4
    assert (out / "Cell_001_dsr.ome.zarr").is_dir()
    cxc = (out / "Cell_001_dsr.cxc").read_text()
    # One vseries per channel, or ChimeraX opens 100 separate models.
    assert cxc.count("vseries true") == 2
    assert json.loads((out / "viewer_export.json").read_text())["voxel_um"] == DSR_VOXEL_UM
    # And no half-written temporaries left behind.
    assert not list(src.glob("*.tmp*")) and not list(out.glob("*.tmp"))


def _make_single_timepoint(d, names=("NewDay_006_GFP_488.ome", "NewDay_006_mScarlet_561.ome"),
                           shape=(6, 8, 8)):
    """A single-timepoint zarr dataset's DSR output: one frame per channel,
    named after its store, plus a MIP that must be ignored."""
    rng = np.random.default_rng(1)
    for n in names:
        # ome=False: tifffile would otherwise add OME-XML on its own for an
        # `.ome.tif` name, and PetaKit5D's writetiff never does.
        tifffile.imwrite(d / f"{n}.tif", (rng.random(shape) * 1000).astype(np.uint16),
                         compression="lzw", ome=False)
    tifffile.imwrite(d / f"{names[0]}_MIP_z.tif", np.zeros((4, 4), np.uint16))
    return list(names)


def test_parse_frames_single_timepoint_by_store_name(tmp_path):
    """Rig 2026-09-23: single-timepoint streamed datasets matched no frames,
    so they never got a viewer export at all."""
    names = _make_single_timepoint(tmp_path)
    assert parse_dsr_frames(tmp_path) == {}
    frames = parse_dsr_frames(tmp_path, names)
    assert frames == {(0, 0): tmp_path / f"{names[0]}.tif", (1, 0): tmp_path / f"{names[1]}.tif"}


def test_export_single_timepoint_both(tmp_path):
    zarr = pytest.importorskip("zarr")
    src = tmp_path / "dsr"
    src.mkdir()
    names = _make_single_timepoint(src)
    out = tmp_path / "viewer"
    summary = export_for_viewers(src, out, name="NewDay_006", single_names=names,
                                 channel_labels=["GFP 488", "mScarlet 561"])
    assert summary["frames"] == 2 and summary["stamped"] == 2
    g = zarr.open_group(str(out / "NewDay_006_dsr.ome.zarr"), mode="r")
    assert g["0"].shape == (1, 2, 6, 8, 8)
    cxc = (out / "NewDay_006_dsr.cxc").read_text()
    assert f'open "{(src / f"{names[0]}.tif").resolve()}"' in cxc
    assert "vseries" not in cxc


@pytest.mark.parametrize("single", [False, True])
def test_export_ome_zarr_only_replaces_tiffs_after_verifying(tmp_path, single):
    zarr = pytest.importorskip("zarr")
    src = tmp_path / "dsr"
    src.mkdir()
    if single:
        names = _make_single_timepoint(src)
    else:
        names = None
        _make_frames(src, channels=2, times=2, shape=(6, 8, 8))
    originals = {k: tifffile.imread(p) for k, p in parse_dsr_frames(src, names).items()}

    summary = export_for_viewers(src, tmp_path / "viewer", name="X",
                                 single_names=names, output_format="ome-zarr")

    assert summary["removed_tiffs"] == len(originals)
    assert parse_dsr_frames(src, names) == {}
    # PetaKit5D's MIPs are not ours to delete.
    if single:
        assert (src / f"{names[0]}_MIP_z.tif").is_file()
    g = zarr.open_group(str(tmp_path / "viewer" / "X_dsr.ome.zarr"), mode="r")
    times = sorted({t for _, t in originals})
    for (c, t), vol in originals.items():
        np.testing.assert_array_equal(g["0"][times.index(t), c], vol)
    assert "chimerax" not in summary


def test_export_tiff_only_skips_zarr(tmp_path):
    src = tmp_path / "dsr"
    src.mkdir()
    _make_frames(src, channels=1, times=2, shape=(6, 8, 8))
    out = tmp_path / "viewer"
    summary = export_for_viewers(src, out, name="X", output_format="tiff")
    assert summary["stamped"] == 2 and "ome_zarr" not in summary
    assert not (out / "X_dsr.ome.zarr").exists()
    assert len(parse_dsr_frames(src)) == 2


def test_remove_frames_keeps_tiffs_on_mismatch(tmp_path):
    pytest.importorskip("zarr")
    from backfill.viewer_export import remove_frames_verified

    src = tmp_path / "dsr"
    src.mkdir()
    _make_frames(src, channels=1, times=2, shape=(6, 8, 8))
    out = write_ome_zarr(src, tmp_path / "x.ome.zarr", levels=1)
    frames = parse_dsr_frames(src)
    tifffile.imwrite(frames[(0, 1)], np.ones((6, 8, 8), np.uint16))  # diverge after export
    with pytest.raises(RuntimeError, match="does not match"):
        remove_frames_verified(frames, out)
    assert frames[(0, 1)].is_file()


def test_export_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="output_format"):
        export_for_viewers(tmp_path, tmp_path / "v", name="X", output_format="png")


# --- reusing a store the live lane (or an earlier run) already built -----


def _live_built_store(tmp_path, frames_dir, n_t=3, n_c=2, shape=(8, 12, 10)):
    """What opym.stream.live leaves: the export's own path, filled from the
    same frames, progress marked complete after the last frame."""
    import os
    import time

    from opym import ome_zarr_writer as w

    out_dir = tmp_path / "viewer"
    store = out_dir / "Cell_002_dsr.ome.zarr"
    w.create_store(store, n_t=n_t, n_c=n_c, shape_zyx=shape, dtype=np.uint16)
    for t in range(n_t):
        for c in range(n_c):
            f = frames_dir / f"Cell_001_C{c}_T{t:03d}.tif"
            if f.exists():
                w.write_timepoint(store, t, c, tifffile.imread(f))
    done = [[t, c] for t in range(n_t) for c in range(n_c)]
    old = time.time() - 60
    for f in frames_dir.glob("*.tif"):
        os.utime(f, (old, old))
    w.write_progress(store, n_t=n_t, n_c=n_c, done=done, state="complete")
    return out_dir, store


def test_export_reuses_a_complete_live_store(tmp_path):
    d = tmp_path / "dsr"
    d.mkdir()
    _make_frames(d)
    out_dir, store = _live_built_store(tmp_path, d)
    (store / "0" / "live_marker").write_text("untouched")

    summary = export_for_viewers(d, out_dir, name="Cell_002")
    assert summary.get("ome_zarr_reused") is True
    assert (store / "0" / "live_marker").exists(), "store was rebuilt"


def test_export_rebuilds_when_a_frame_is_newer_than_the_store(tmp_path):
    d = tmp_path / "dsr"
    d.mkdir()
    _make_frames(d)
    out_dir, store = _live_built_store(tmp_path, d)
    (store / "0" / "live_marker").write_text("stale")
    import os
    import time

    newer = time.time() + 5
    os.utime(next(d.glob("*_C0_T001.tif")), (newer, newer))

    summary = export_for_viewers(d, out_dir, name="Cell_002")
    assert not summary.get("ome_zarr_reused")
    assert not (store / "0" / "live_marker").exists()


def test_export_rebuilds_an_aborted_acquisitions_oversized_store(tmp_path):
    """Live allocates the declared timepoint count; an aborted acquisition has
    fewer frames, and the export's store must match what exists."""
    d = tmp_path / "dsr"
    d.mkdir()
    _make_frames(d, times=2)
    out_dir, store = _live_built_store(tmp_path, d, n_t=5)

    summary = export_for_viewers(d, out_dir, name="Cell_002")
    assert not summary.get("ome_zarr_reused")
    import zarr

    assert zarr.open_group(str(store), mode="r")["0"].shape[0] == 2
