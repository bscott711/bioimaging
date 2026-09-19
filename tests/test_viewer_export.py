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
