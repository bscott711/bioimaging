#!/usr/bin/env python3
"""Residual-haze background term for decon, applied at staging.

A PSF (even the long-tailed synthetic one) cannot account for light that is smooth over many um: scatter, and fluorescence
from outside the imaged slab. This estimates that floor as the LOWER ENVELOPE of the staged volume (grey opening with a
footprint wider than the membrane, then smoothed) and subtracts it before decon, so the RL/OMW iterations start from
cell signal over a flat floor instead of cell signal on a pedestal. The envelope can only lie at or below the data, and the
camera offset is kept, so the result is never below `offset`.

Volume layout is the staged (ny, nx, nz): y, x at `xy_um`, z at the scan step `dz_um` (skewed space).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CAMERA_OFFSET = 100.0  # counts, exactly 100 on this camera (memory: project_decon_reenable)


def haze_floor(vol: np.ndarray, xy_um: float = 0.136, dz_um: float = 0.5, open_um: float = 3.0,
               smooth_um: float = 3.0, ds: int = 4) -> np.ndarray:
    """Smooth lower envelope of ``vol`` (same shape), float32, computed on a ``ds``-times coarser lateral grid."""
    v = vol.astype(np.float32)
    ny, nx, nz = v.shape
    small = ndi.uniform_filter(v, size=(ds, ds, 1))[::ds, ::ds, :]
    box = (max(int(round(open_um / (xy_um * ds))) | 1, 3), max(int(round(open_um / (xy_um * ds))) | 1, 3),
           max(int(round(open_um / dz_um)) | 1, 3))
    env = ndi.maximum_filter(ndi.minimum_filter(small, size=box), size=box)
    env = ndi.gaussian_filter(env, (smooth_um / (xy_um * ds), smooth_um / (xy_um * ds), smooth_um / dz_um))
    zoom = (ny / env.shape[0], nx / env.shape[1], 1.0)
    return np.minimum(ndi.zoom(env, zoom, order=1)[:ny, :nx, :nz], v).astype(np.float32)


def subtract_haze(vol: np.ndarray, gamma: float = 1.0, offset: float = CAMERA_OFFSET, **kw) -> np.ndarray:
    """``vol`` minus ``gamma`` times the haze floor above the camera offset; never below ``offset``'s share of the input."""
    floor = np.clip(haze_floor(vol, **kw) - offset, 0, None)
    out = vol.astype(np.float32) - gamma * floor
    return np.clip(out, 0, np.iinfo(np.uint16).max).astype(np.uint16)
