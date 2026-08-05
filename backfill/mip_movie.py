"""Encodes the per-timepoint Z-MIP TIFFs PetaKit5D writes (once `save_mip`
is enabled, see opym's run_petakit_server.m) into browsable MP4 movies for
the Argus dashboard's MIP browser.

No movie-encoding code existed anywhere in these repos before this --
`bioimaging/scripts/export_mip_tif.py` is the closest precedent (globs a
`Decon/` dir's per-T,C zarr stores and writes one static ImageJ-hyperstack
TIFF per channel), reused here for its glob/group-by-channel pattern only;
this module's input is already-2D MIP TIFFs (PetaKit5D did the Z-max
itself) and its output is an actual video, not a static TIFF stack.
"""

from __future__ import annotations

import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

_MIP_RE = re.compile(r"_C(\d+)_T(\d+)_MIP_z\.tif$")

# Without this, ffmpeg writes the `moov` atom (the container's format/seek
# index -- what a browser needs before it can even recognize the file as
# playable video) after all the frame data instead of before it. A local
# player that reads the whole file is fine either way, but the dashboard
# serves these over HTTP with Range support for <video> scrubbing, and a
# browser's initial (partial) fetch never reaches a trailing moov atom --
# it just rejects the file outright ("no supported format"). This single
# flag moves moov to the front of the file.
_FASTSTART_PARAMS = ["-movflags", "+faststart"]

# Fixed per-channel pseudo-colors for the additive composite view (RGB,
# 0-1 floats) -- cyan/magenta/yellow/red covers the 4-channel-per-excitation
# output map (see core.py: 0=Bot-C0, 1=Top-C0, 2=Top-C1, 3=Bot-C1) with
# maximally distinguishable hues; extra channels beyond 4 cycle back.
_CHANNEL_COLORS = [
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.0, 1.0),  # magenta
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.15, 0.15),  # red
]


def find_mip_files(mips_dir: Path) -> dict[int, list[tuple[int, Path]]]:
    """Groups `<name>_C{c}_T{t}_MIP_z.tif` files by channel, sorted by T."""
    by_channel: dict[int, list[tuple[int, Path]]] = {}
    if not mips_dir.is_dir():
        return by_channel
    for f in mips_dir.glob("*_MIP_z.tif"):
        m = _MIP_RE.search(f.name)
        if not m:
            continue
        c, t = int(m.group(1)), int(m.group(2))
        by_channel.setdefault(c, []).append((t, f))
    for files in by_channel.values():
        files.sort(key=lambda pair: pair[0])
    return by_channel


def load_channel_stack(files: list[tuple[int, Path]]) -> np.ndarray:
    """(T, Y, X) stack -- these MIP files are already 2D (PetaKit5D did the
    Z-max), so no further projection is needed here."""
    frames = [tifffile.imread(p) for _t, p in files]
    return np.stack(frames, axis=0)


def normalize_for_video(
    stack: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.8
) -> np.ndarray:
    """Percentile-stretches to uint8 using ONE shared (low, high) computed
    across the whole T-stack (not per-frame), so brightness doesn't flicker
    frame-to-frame during playback.
    """
    lo, hi = np.percentile(stack, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(stack, dtype=np.uint8)
    scaled = (stack.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def encode_channel_movie(stack_u8: np.ndarray, out_path: Path, fps: float = 12.0) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path, stack_u8, fps=fps, codec="libx264", pixelformat="yuv420p",
        output_params=_FASTSTART_PARAMS,
    )
    return out_path


def encode_composite_movie(
    channel_stacks: dict[int, np.ndarray], out_path: Path, fps: float = 12.0
) -> Path:
    """Additively blends per-channel-normalized uint8 stacks into a single
    pseudo-colored RGB movie -- the default triage view."""
    channels = sorted(channel_stacks)
    t_len = next(iter(channel_stacks.values())).shape[0]
    shape_yx = next(iter(channel_stacks.values())).shape[1:]

    composite = np.zeros((t_len, *shape_yx, 3), dtype=np.float32)
    for i, c in enumerate(channels):
        color = np.array(_CHANNEL_COLORS[i % len(_CHANNEL_COLORS)], dtype=np.float32)
        gray = channel_stacks[c].astype(np.float32) / 255.0  # (T, Y, X)
        composite += gray[..., None] * color

    composite_u8 = np.clip(composite, 0, 1.0)
    composite_u8 = (composite_u8 * 255.0).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path, composite_u8, fps=fps, codec="libx264", pixelformat="yuv420p",
        output_params=_FASTSTART_PARAMS,
    )
    return out_path


def encode_poster_image(frame_u8: np.ndarray, out_path: Path) -> Path:
    """A static first-frame JPEG alongside each movie. `<video>` shows a
    black box until it has enough data to paint a frame -- true even with
    faststart, and doubly true if a browser has throttled/deferred autoplay
    (which it will, once a page has hundreds of thumbnails). An explicit
    `poster=` is the standard fix: something meaningful renders immediately,
    independent of whether/when the video itself starts playing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frame_u8, extension=".jpg")
    return out_path


def build_mip_movies_for_dataset(
    dsr_dir: Path, sanitized_name: str, out_dir: Path, fps: float = 12.0
) -> list[Path]:
    """Orchestrates the above for one dataset. Writes to
    `<leaf_dir>/mip_movies/` -- a flat, crop-format-agnostic location (NOT
    nested under `DSR_nodecon/MIPs/`) so the registry and the dashboard can
    find every dataset's movies via one predictable glob regardless of
    internal crop-stage layout. Each movie gets a same-named `.jpg` poster
    (`<name>.mp4` -> `<name>.jpg`) for the dashboard's thumbnail grid.
    """
    mips_dir = dsr_dir / "MIPs"
    by_channel = find_mip_files(mips_dir)
    if not by_channel:
        raise FileNotFoundError(f"No MIP TIFFs found under {mips_dir}")

    written: list[Path] = []
    normalized_stacks: dict[int, np.ndarray] = {}
    for c, files in by_channel.items():
        stack = load_channel_stack(files)
        stack_u8 = normalize_for_video(stack)
        normalized_stacks[c] = stack_u8
        out_path = out_dir / f"{sanitized_name}_C{c}.mp4"
        written.append(encode_channel_movie(stack_u8, out_path, fps=fps))
        written.append(encode_poster_image(stack_u8[0], out_path.with_suffix(".jpg")))

    if len(normalized_stacks) > 1:
        composite_path = out_dir / f"{sanitized_name}_composite.mp4"
        written.append(encode_composite_movie(normalized_stacks, composite_path, fps=fps))
        # Recompute frame 0 of the composite blend for its poster (cheap --
        # one frame -- rather than threading it back out of
        # encode_composite_movie's internals).
        first_frame = np.zeros((*normalized_stacks[next(iter(normalized_stacks))].shape[1:], 3), dtype=np.float32)
        for i, c in enumerate(sorted(normalized_stacks)):
            color = np.array(_CHANNEL_COLORS[i % len(_CHANNEL_COLORS)], dtype=np.float32)
            first_frame += (normalized_stacks[c][0].astype(np.float32) / 255.0)[..., None] * color
        first_frame_u8 = (np.clip(first_frame, 0, 1.0) * 255.0).astype(np.uint8)
        written.append(encode_poster_image(first_frame_u8, composite_path.with_suffix(".jpg")))

    return written
