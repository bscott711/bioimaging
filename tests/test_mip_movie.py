"""Contract for `load_channel_stack`'s stale-frame tolerance.

Pure Python -- writes tiny fake MIP TIFFs to tmp_path, no MATLAB, no GPU,
no real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from backfill.mip_movie import _channel_color, load_channel_stack, write_composite


def test_stale_mismatched_frame_is_dropped_and_deleted(tmp_path):
    """Regression for a real failure (`.../20260304_YGbeads_30PDMS_AH/dhDF_2`):
    one MIP left over from a run with different crop/deskew geometry sat
    alongside a full run of correctly-sized frames, and `np.stack` raised
    "all input arrays must have the same shape". The stale frame must be
    dropped from the stack (not just from the count) and removed from disk
    so it isn't rediscovered on a later pass.
    """
    files: list[tuple[int, Path]] = []
    for t in range(3):
        p = tmp_path / f"x_C0_T{t:04d}_MIP_z.tif"
        tifffile.imwrite(p, np.full((10, 12), t, dtype=np.uint16))
        files.append((t, p))
    stale = tmp_path / "x_C0_T0003_MIP_z.tif"
    tifffile.imwrite(stale, np.zeros((7, 8), dtype=np.uint16))  # wrong shape
    files.append((3, stale))

    stack = load_channel_stack(files)

    assert stack.shape == (3, 10, 12)
    assert not stale.exists(), "the stale mismatched frame should be removed from disk"


def test_all_matching_frames_are_kept(tmp_path):
    files: list[tuple[int, Path]] = []
    for t in range(4):
        p = tmp_path / f"x_C0_T{t:04d}_MIP_z.tif"
        tifffile.imwrite(p, np.full((10, 12), t, dtype=np.uint16))
        files.append((t, p))

    stack = load_channel_stack(files)

    assert stack.shape == (4, 10, 12)
    for t in range(4):
        assert files[t][1].exists()


def test_composite_channel_two_is_magenta_never_red():
    """Channel 1 (the second) is magenta in every composite; the palette is opym's."""
    assert tuple(_channel_color(0)) == (0.0, 1.0, 0.0)
    assert tuple(_channel_color(1)) == (1.0, 0.0, 1.0)
    assert tuple(_channel_color(4)) == tuple(_channel_color(0)), "wraps past the fourth"


def test_composite_poster_of_two_channels_is_green_and_magenta(tmp_path):
    """Channel 0 alone lights green only, channel 1 alone lights red+blue (magenta)."""
    import imageio.v3 as iio

    bright = np.full((1, 8, 8), 255, np.uint8)
    dark = np.zeros((1, 8, 8), np.uint8)
    for stacks, expect in (({0: bright, 1: dark}, (0, 255, 0)), ({0: dark, 1: bright}, (255, 0, 255))):
        out = tmp_path / f"c{expect[0]}.webm"
        poster = write_composite(stacks, out)[1]
        px = iio.imread(poster)[4, 4]
        assert all(abs(int(a) - b) <= 12 for a, b in zip(px, expect)), (px, expect)
