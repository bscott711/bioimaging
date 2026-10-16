"""Camera bias and haze floor: the floor subtracts smooth pedestal, keeps thin bright structure, never exceeds the data;
the bias is subtracted in float without a clamp; the axis layout is explicit and physically right on every axis."""
import numpy as np
import pytest

from psf_tools.decon_haze_bg import estimate_dark_map, haze_floor, subtract_bias, subtract_haze

SHAPE = (80, 60, 40)  # (ny, nx, nz)
YXZ = "yxz"


def _phantom():
    rng = np.random.default_rng(0)
    y, x, z = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    haze = 100 + 60 + 0.8 * y  # camera offset + a smooth ramp
    sheet = np.where(x == 30, 200.0, 0.0)  # a one-voxel-thick membrane
    return (haze + sheet + rng.normal(0, 2, SHAPE)).astype(np.uint16), sheet


def test_floor_never_exceeds_the_data_and_follows_the_ramp():
    vol, _ = _phantom()
    f = haze_floor(vol, axes=YXZ, xy_um=0.5, dz_um=0.5, open_um=3.0, smooth_um=2.0, ds=2)
    assert (f <= vol + 1e-3).all()
    assert abs(float(np.median(f[:, 5, :]) - np.median(vol[:, 5, :]))) < 20  # tracks the ramp where there is no sheet


def test_subtraction_removes_the_pedestal_but_keeps_the_sheet():
    vol, sheet = _phantom()
    out = subtract_haze(vol, axes=YXZ, xy_um=0.5, dz_um=0.5, open_um=3.0, smooth_um=2.0, ds=2)
    off = np.median(out[:, 5, :].astype(float))
    on = np.median(out[:, 30, :].astype(float))
    assert abs(off - 100) < 6                        # pedestal gone, camera offset (100) kept
    assert on - off > 0.9 * 200                      # the 200-count sheet survives (opening footprint 3 um >> 1 voxel)
    assert (out <= vol).all()


def test_a_footprint_that_fits_inside_the_object_eats_it():
    """Negative control for the footprint choice: a bright slab 20 voxels wide survives a footprint wider than it, but a
    footprint that fits inside the slab sees it as floor and subtracts it (why `open_um` must exceed the membrane's width)."""
    vol = np.full(SHAPE, 160, np.uint16)
    vol[:, 20:40, :] = 360
    wide = subtract_haze(vol, axes=YXZ, xy_um=0.5, dz_um=0.5, open_um=30.0, smooth_um=1.0, ds=2)
    narrow = subtract_haze(vol, axes=YXZ, xy_um=0.5, dz_um=0.5, open_um=3.0, smooth_um=1.0, ds=2)
    assert wide[:, 30, :].astype(float).mean() > 280    # kept on top of the haze
    assert narrow[:, 30, :].astype(float).mean() < 130  # eaten


def test_subtraction_is_the_data_minus_the_floor_above_the_offset():
    """What `stage_bg` removes is exactly the haze floor's excess over the camera offset (so the floor view shows what is taken)."""
    vol, _ = _phantom()
    kw = dict(axes=YXZ, xy_um=0.5, dz_um=0.5, open_um=3.0, smooth_um=2.0, ds=2)
    floor = haze_floor(vol, **kw)
    out = subtract_haze(vol, **kw)
    expected = np.clip(vol.astype(np.float32) - np.clip(floor - 100.0, 0, None), 0, 65535)
    assert np.abs(out.astype(np.float32) - expected).max() <= 1.0   # uint16 rounding only


# ---- axis layout -----------------------------------------------------------------------------------------------------

def _slab(axis: str, lo_um: float, hi_um: float):
    """Staged-layout (nz, ny, nx) volume: 150-count haze everywhere plus a +200 slab between ``lo_um`` and ``hi_um`` along
    ``axis`` (z is sampled at 0.5 um, y and x at 0.136 um), and the slab's centre index."""
    step = {"z": 0.5, "y": 0.136, "x": 0.136}
    vol = np.full((44, 120, 120), 150.0, np.float32)
    i = "zyx".index(axis)
    lo, hi = int(round(lo_um / step[axis])), int(round(hi_um / step[axis]))
    idx = [slice(None)] * 3
    idx[i] = slice(lo, hi)
    vol[tuple(idx)] += 200.0
    return vol, i, (lo + hi) // 2


@pytest.mark.parametrize("axis", ["z", "y", "x"])
def test_the_three_um_footprint_is_three_um_along_every_axis(axis):
    """A slab 1.5 um thick (thinner than the footprint) is structure and survives; one 8 um thick (wider) is floor and is
    removed. Both must hold along z AND y AND x of the staged (nz, ny, nx) layout. The trial once passed (nz, ny, nx) arrays to
    code that read them as (ny, nx, nz): footprints were ~10 um along the scan axis and <1 um along x, and this fails there."""
    kw = dict(axes="zyx", xy_um=0.136, dz_um=0.5, open_um=3.0, smooth_um=1.0, ds=1)
    for lo, hi, survives in ((3.0, 4.5, True), (3.0, 11.0, False)):
        vol, i, c = _slab(axis, lo, hi)
        out = subtract_haze(vol.astype(np.uint16), **kw).astype(float)
        sl = [10, 60, 60]
        sl[i] = c
        kept = out[tuple(sl)] - 100.0  # the offset (100) stays; what is left above it
        base = [10, 60, 60]
        base[i] = 0
        kept -= out[tuple(base)] - 100.0
        assert (kept > 150.0) if survives else (kept < 60.0), (axis, lo, hi, kept)


def test_axis_labels_are_a_pure_relabelling():
    """The same physical volume in two array orders gives the same floor (up to the transpose)."""
    rng = np.random.default_rng(1)
    zyx = (150 + rng.normal(0, 3, (20, 70, 50))).astype(np.float32)
    zyx[:, 20:30, :] += 80
    kw = dict(xy_um=0.136, dz_um=0.5, open_um=3.0, smooth_um=1.0, ds=2)
    a = haze_floor(zyx, axes="zyx", **kw)
    b = haze_floor(np.transpose(zyx, (1, 2, 0)), axes="yxz", **kw)
    np.testing.assert_allclose(a, np.transpose(b, (2, 0, 1)), atol=1e-4)
    assert a.shape == zyx.shape


@pytest.mark.parametrize("bad", ["zy", "zyxx", "abc", ""])
def test_a_bad_axes_string_is_refused(bad):
    with pytest.raises(ValueError):
        haze_floor(np.zeros((4, 5, 6), np.float32), axes=bad)


def test_axes_has_no_default():
    with pytest.raises(TypeError):
        haze_floor(np.zeros((4, 5, 6), np.float32))  # type: ignore[call-arg]


# ---- camera bias -----------------------------------------------------------------------------------------------------

def _dark_volume(shape=(60, 80, 70), seed=2):
    rng = np.random.default_rng(seed)
    return rng.normal(100.0, 4.5, shape)  # offset 100, read noise sd 4.5 (measured)


def test_bias_subtraction_is_exact_and_not_clamped():
    vol = np.rint(_dark_volume()).astype(np.uint16)
    out = subtract_bias(vol, axes="zyx")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, vol.astype(np.float32) - 100.0)
    assert (out < 0).mean() > 0.4 and abs(out.mean()) < 0.1  # noise stays zero-mean


def test_clamping_after_the_subtraction_biases_dark_regions_up():
    """Why the bias is not clamped before N2V: max(noise, 0) has mean 0.4 sd = +1.8 counts at sd 4.5, the order of a dim
    cell's signal above the offset. This is the number measured on the real volumes (+1.6 to +2.1)."""
    vol = np.rint(_dark_volume((80, 120, 120))).astype(np.uint16)
    free = subtract_bias(vol, axes="zyx")
    clamped = np.clip(free, 0, None)
    assert abs(free.mean()) < 0.1
    assert 1.5 < clamped.mean() < 2.1


def test_dark_map_recovers_a_fixed_pattern_that_every_plane_shares():
    rng = np.random.default_rng(3)
    ny, nx = 90, 70
    pattern = rng.normal(0, 0.9, (ny, 1)) + rng.normal(0, 0.3, (1, nx)) + rng.normal(0, 0.3, (ny, nx))  # mostly row-wise
    vols = []
    for s in range(3):
        v = rng.normal(100.0, 4.5, (80, ny, nx)) + pattern[None]
        v[:, 35:55, 25:45] += 60  # a bright cell: those pixels never see dark voxels, so they get no correction
        vols.append(v.astype(np.float32))
    dmap, cnt = estimate_dark_map(vols, axes="zyx")
    seen = cnt >= 60
    assert not seen[40:50, 30:40].any() and (dmap[~seen] == 0).all()   # no information -> no correction
    assert np.corrcoef(dmap[seen], pattern[seen])[0, 1] > 0.9
    # removing it flattens the per-pixel z-mean of a fresh dark volume (pattern sd 0.9 -> noise-limited ~0.3)
    fresh = rng.normal(100.0, 4.5, (80, ny, nx)) + pattern[None]
    before = fresh.mean(0)[seen].std()
    after = subtract_bias(fresh.astype(np.float32), axes="zyx", dark_map=dmap, keep_offset=True).mean(0)[seen].std()
    assert after < 0.6 * before
    assert abs(subtract_bias(fresh.astype(np.float32), axes="zyx", dark_map=dmap, keep_offset=True).mean() - 100) < 0.5


def test_dark_map_must_match_the_camera_frame():
    with pytest.raises(ValueError):
        subtract_bias(np.zeros((5, 6, 7), np.float32), axes="zyx", dark_map=np.zeros((7, 6), np.float32))
