"""The haze floor subtracts smooth pedestal, keeps a thin bright sheet, and never exceeds the data."""
import numpy as np

from psf_tools.decon_haze_bg import haze_floor, subtract_haze

SHAPE = (80, 60, 40)  # (ny, nx, nz)


def _phantom():
    rng = np.random.default_rng(0)
    y, x, z = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    haze = 100 + 60 + 0.8 * y  # camera offset + a smooth ramp
    sheet = np.where(x == 30, 200.0, 0.0)  # a one-voxel-thick membrane
    return (haze + sheet + rng.normal(0, 2, SHAPE)).astype(np.uint16), sheet


def test_floor_never_exceeds_the_data_and_follows_the_ramp():
    vol, _ = _phantom()
    f = haze_floor(vol, xy_um=0.5, dz_um=0.5, open_um=3.0, smooth_um=2.0, ds=2)
    assert (f <= vol + 1e-3).all()
    assert abs(float(np.median(f[:, 5, :]) - np.median(vol[:, 5, :]))) < 20  # tracks the ramp where there is no sheet


def test_subtraction_removes_the_pedestal_but_keeps_the_sheet():
    vol, sheet = _phantom()
    out = subtract_haze(vol, xy_um=0.5, dz_um=0.5, open_um=3.0, smooth_um=2.0, ds=2)
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
    wide = subtract_haze(vol, xy_um=0.5, dz_um=0.5, open_um=30.0, smooth_um=1.0, ds=2)
    narrow = subtract_haze(vol, xy_um=0.5, dz_um=0.5, open_um=3.0, smooth_um=1.0, ds=2)
    assert wide[:, 30, :].astype(float).mean() > 280    # kept on top of the haze
    assert narrow[:, 30, :].astype(float).mean() < 130  # eaten
