#!/usr/bin/env python3
"""Camera bias and residual-haze background terms for decon, applied at staging.

Two different things live under "background" and they are handled differently (measured, see the decon v2 notebook):

* CAMERA BIAS: a constant offset (exactly 100 counts on this camera, read noise sd ~4.5 counts) plus a small static pattern fixed
  in the camera frame (~0.9 counts rms: a smooth +-1 count profile along the staged y axis plus per-pixel structure; reproducible
  between plane halves, r 0.7, and between cells, r 0.7-0.8; different for the two channels, r 0.25). Bias carries no shot noise.
  The constant is ALREADY subtracted by decon: PetaKit's XR_decon_data_wrapper turns an empty `background` into 100
  (XR_decon_data_wrapper.m:219) and the decon function clamps at 0 (production and the trial's `prod` arm are bit-identical to an
  explicit `background=100`). What is NOT handled is the static pattern. It is the same in every z plane, so N2V's 3-D blind
  spot reads it as signal and passes it through (measured, Cell_004 C1: the pattern's per-pixel sd is 1.09 raw, 0.74 after N2V
  on raw frames, 0.17 with the dark map removed first; the same holds in 5 of the 6 trial volumes, the exception being the bright
  Cell_001_002 C0), so `subtract_bias` with a dark map goes in BEFORE N2V and keeps the
  offset (`keep_offset=True`) for PetaKit to take. The clamp inside decon adds ~+1.7 counts of positive bias to dark voxels
  (noise sd 4.5, clamped at 0); N2V output, which is nearly noise-free, is barely affected.
* HAZE FLOOR: a smooth light floor (scatter, out-of-slab fluorescence). It is PHOTONS, not bias: noise variance rises ~3.0
  counts^2 per count above the offset in every cell, from +2 to +400 counts. It is a modelling choice (a background term in
  y = Hx + b), estimated as the LOWER ENVELOPE of the volume (grey opening with a footprint wider than the membrane, then
  smoothed). N2V leaves smooth structure alone and a lower envelope of noisy data sits ~3 counts low, so the floor is
  estimated AFTER N2V. The envelope can only lie at or below the data and the camera offset is kept.

Every function takes `axes`, a string naming the array axes (for example "zyx" for the staged TIFFs, "yxz" for the PetaKit
layout). It has no default: the trial once passed (nz, ny, nx) arrays to code that assumed (ny, nx, nz), which silently made
the 3 um footprint ~10 um along the scan axis and under 1 um along x.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CAMERA_OFFSET = 100.0  # counts, exactly 100 on this camera (memory: project_decon_reenable)


def _to_yxz(vol: np.ndarray, axes: str) -> tuple[np.ndarray, list[int]]:
    """View of ``vol`` as (y, x, z), and the permutation that undoes it."""
    if len(axes) != 3 or sorted(axes) != ["x", "y", "z"] or vol.ndim != 3:
        raise ValueError(f"axes must be a permutation of 'xyz' matching a 3-D volume, got {axes!r} for ndim {vol.ndim}")
    order = [axes.index(a) for a in "yxz"]
    return np.transpose(vol, order), [int(i) for i in np.argsort(order)]


def haze_floor(vol: np.ndarray, *, axes: str, xy_um: float = 0.136, dz_um: float = 0.5, open_um: float = 3.0,
               smooth_um: float = 3.0, ds: int = 4) -> np.ndarray:
    """Smooth lower envelope of ``vol`` (same shape and axis order), float32, computed on a ``ds``-times coarser lateral grid.
    y, x are at ``xy_um``, z is the scan step ``dz_um`` (skewed space)."""
    v, undo = _to_yxz(vol, axes)
    v = v.astype(np.float32)
    ny, nx, nz = v.shape
    small = ndi.uniform_filter(v, size=(ds, ds, 1))[::ds, ::ds, :]
    box = (max(int(round(open_um / (xy_um * ds))) | 1, 3), max(int(round(open_um / (xy_um * ds))) | 1, 3),
           max(int(round(open_um / dz_um)) | 1, 3))
    env = ndi.maximum_filter(ndi.minimum_filter(small, size=box), size=box)
    env = ndi.gaussian_filter(env, (smooth_um / (xy_um * ds), smooth_um / (xy_um * ds), smooth_um / dz_um))
    zoom = (ny / env.shape[0], nx / env.shape[1], 1.0)
    floor = np.minimum(ndi.zoom(env, zoom, order=1)[:ny, :nx, :nz], v).astype(np.float32)
    return np.ascontiguousarray(np.transpose(floor, undo))


def subtract_haze(vol: np.ndarray, *, axes: str, gamma: float = 1.0, offset: float = CAMERA_OFFSET, **kw) -> np.ndarray:
    """``vol`` minus ``gamma`` times the haze floor above the camera offset, as uint16; the offset itself is kept, so the
    result is never below ``offset``'s share of the input."""
    floor = np.clip(haze_floor(vol, axes=axes, **kw) - offset, 0, None)
    out = vol.astype(np.float32) - gamma * floor
    return np.clip(out, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def estimate_dark_map(vols: list[np.ndarray], *, axes: str = "zyx", offset: float = CAMERA_OFFSET, dark_below: float = 3.0,
                      win: int = 9, min_count: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Per camera pixel (y, x), the mean deviation from ``offset`` over the voxels that are dark (local ``win`` x ``win``
    in-plane mean below ``offset + dark_below``), pooled over ``vols`` (volumes cropped to the same camera pixels).
    Returns ``(dark_map, count)``; pixels with fewer than ``min_count`` dark voxels get 0 (no correction, no information).
    Pooling over cells shrinks the estimate's own noise (read variance ~20 / count)."""
    total = cnt = None
    for vol in vols:
        v, _ = _to_yxz(vol, axes)
        v = v.astype(np.float32)
        dark = ndi.uniform_filter(v, size=(win, win, 1)) < offset + dark_below
        s = (v * dark).sum(2)
        c = dark.sum(2)
        total = s if total is None else total + s
        cnt = c if cnt is None else cnt + c
    ok = cnt >= min_count
    dmap = np.where(ok, total / np.maximum(cnt, 1) - offset, 0.0).astype(np.float32)
    return dmap, cnt.astype(np.int32)


def subtract_bias(vol: np.ndarray, *, axes: str, offset: float = CAMERA_OFFSET, dark_map: np.ndarray | None = None,
                  keep_offset: bool = False) -> np.ndarray:
    """Camera bias out of ``vol``, in float32 and NOT clamped, so zero-mean noise stays zero-mean.
    ``dark_map`` is the per-camera-pixel (y, x) deviation from ``offset`` (see `estimate_dark_map`); with
    ``keep_offset`` only the pattern is removed and the constant offset stays (for stages that feed PetaKit, which
    subtracts the scalar itself through its ``background`` setting)."""
    out = vol.astype(np.float32)
    if not keep_offset:
        out = out - np.float32(offset)
    if dark_map is not None:
        d, undo = _to_yxz(np.zeros(vol.shape, np.float32), axes)
        ny, nx, _ = d.shape
        if dark_map.shape != (ny, nx):
            raise ValueError(f"dark_map {dark_map.shape} does not match the volume's camera frame {(ny, nx)}")
        out = out - np.transpose(np.broadcast_to(dark_map[:, :, None], d.shape), undo)
    return out
