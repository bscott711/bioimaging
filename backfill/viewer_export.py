"""Make DSR output open directly in napari and ChimeraX, with real units.

PetaKit5D writes the DSR result as a bare TIFF: no resolution tags, no OME
block, nothing (verified on a real Cell_002 DSR frame -- `imagej_metadata`
None, `XResolution` None). Both viewers therefore show it at 1 px per unit,
and anisotropy that isn't there appears the moment you rotate the volume.

The two viewers need different things, which was established by testing them
rather than assuming:

  * ChimeraX reads voxel size straight out of an OME-TIFF -- opening one gives
    step=(0.136, 0.136, 0.136) exactly. (It also reads ImageJ `spacing`, but
    via a rational tag, so it arrives as 0.13599999997.) So the DSR TIFFs are
    restamped in place as OME-TIFF.
  * napari's builtin TIFF reader ignores voxel size entirely -- scale comes
    back (1, 1, 1) for OME-TIFF *and* ImageJ TIFF alike. It does honour
    OME-Zarr through napari-ome-zarr, which returns scale
    (1, 0.136, 0.136, 0.136) and splits channels into named layers. So a
    pyramidal OME-Zarr is written alongside.

Axis order: the DSR TIFF's pages are PetaKit5D's 3rd dimension, so numpy sees
(z, y, x) -- z the optical axis after rotation, y the coverslip long axis.
DSR resamples to an isotropic grid, so all three share one voxel size.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import tifffile

# DSR resamples onto an isotropic grid whose spacing is the lateral pixel
# size; see the xyPixelSize passed in submit_zarr_deskew_ticket.
DSR_VOXEL_UM = 0.136

# napari opens a pyramid lazily; a single 419x1458x392 level per timepoint is
# slow to pan at full resolution. Three levels cost ~14% extra storage.
PYRAMID_LEVELS = 3

_FRAME_RE = re.compile(r"^(?P<prefix>.+)_C(?P<c>\d+)_T(?P<t>\d+)\.tif$")

# Distinct, colour-blind-safe-ish emission colours; index = channel number.
_CHANNEL_COLORS = ("00FF00", "FF3D3D", "00B3FF", "FFC400")


OUTPUT_FORMATS = ("tiff", "ome-zarr", "both")
"""What to keep of the DSR result: the stamped OME-TIFF frames, a single
OME-Zarr, or both (the default -- what this module always produced)."""


def parse_dsr_frames(
    dsr_dir: Path, single_names: list[str] | None = None
) -> dict[tuple[int, int], Path]:
    """Map (channel, timepoint) -> DSR frame path.

    Time series frames are `_C<c>_T<t>.tif`. A single-timepoint zarr dataset
    is never exploded into that form, so PetaKit5D names each channel's frame
    after its store instead (`<name>.ome.tif`); pass those stems, in channel
    order, as `single_names` to pick them up as timepoint 0. Without them a
    single-timepoint dataset matched nothing and silently got no export.

    Either way PetaKit5D's own `MIPs/` output and any `*_MIP_z.tif` sitting
    beside the frames are excluded.
    """
    dsr_dir = Path(dsr_dir)
    frames: dict[tuple[int, int], Path] = {}
    for p in sorted(dsr_dir.glob("*.tif")):
        m = _FRAME_RE.match(p.name)
        if m:
            frames[(int(m.group("c")), int(m.group("t")))] = p
    if not frames and single_names:
        for c, name in enumerate(single_names):
            p = dsr_dir / f"{name}.tif"
            if p.is_file():
                frames[(c, 0)] = p
    return frames


def channel_label(zarr_name: str) -> str:
    """`Cell_002_GFP_488.ome.zarr` -> `GFP 488`.

    Falls back to the whole stem when the name doesn't carry a
    `<dataset>_<fluor>_<wavelength>` tail, so an unexpected name degrades to
    something readable instead of raising.
    """
    stem = zarr_name.removesuffix(".zarr").removesuffix(".ome")
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return f"{parts[-2]} {parts[-1]}"
    return stem


def stamp_ome_tiff(path: Path, *, voxel_um: float = DSR_VOXEL_UM, channel: str | None = None) -> bool:
    """Rewrite one DSR frame as an OME-TIFF carrying its voxel size.

    Returns False if the file already has an OME block, so a re-run over a
    finished dataset costs one header read per frame instead of a full
    rewrite. Writes `.tmp` then `os.replace`, because a half-written frame
    that `readtiff` chokes on would poison every later pass -- the same
    lesson as _clean_stale_deskew_output.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        if tf.is_ome:
            return False
        data = tf.asarray()
        # Carry the source's compression over. PetaKit5D writes these LZW
        # (tag 5); rewriting them uncompressed took one 85 MB pair to 914 MB,
        # which across ~1800 frames would be tens of terabytes.
        src_compression = tf.pages[0].compression

    meta = {
        "axes": "ZYX",
        "PhysicalSizeX": voxel_um, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": voxel_um, "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": voxel_um, "PhysicalSizeZUnit": "µm",
    }
    if channel:
        meta["Channel"] = {"Name": channel}

    try:
        compression = tifffile.COMPRESSION(src_compression).name
    except ValueError:                      # unknown codec: keep the pixels, drop the codec
        compression = None
    if compression == "NONE":
        compression = None

    tmp = path.with_suffix(".tmp.ome.tif")
    tifffile.imwrite(tmp, data, ome=True, metadata=meta, compression=compression,
                     resolution=(1.0 / voxel_um, 1.0 / voxel_um),
                     resolutionunit="MICROMETER")
    os.replace(tmp, path)
    return True


def _downsample(vol: np.ndarray) -> np.ndarray:
    """2x block mean on each axis, trimming any odd trailing plane/row/column."""
    z, y, x = (s - (s % 2) for s in vol.shape)
    v = vol[:z, :y, :x].astype(np.float32)
    v = v.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))
    return v.astype(vol.dtype)


def write_ome_zarr(
    dsr_dir: Path,
    out_path: Path,
    *,
    voxel_um: float = DSR_VOXEL_UM,
    channel_labels: list[str] | None = None,
    levels: int = PYRAMID_LEVELS,
    time_interval_s: float = 1.0,
    single_names: list[str] | None = None,
) -> Path:
    """Assemble the DSR frame series into one pyramidal (t, c, z, y, x) NGFF store.

    One frame is held in memory at a time: the full series is ~9 GB per
    dataset, and the backfill fans several datasets out across a process pool.
    """
    import zarr

    dsr_dir, out_path = Path(dsr_dir), Path(out_path)
    frames = parse_dsr_frames(dsr_dir, single_names)
    if not frames:
        raise FileNotFoundError(f"no DSR frames under {dsr_dir}")

    channels = sorted({c for c, _ in frames})
    times = sorted({t for _, t in frames})
    nz, ny, nx = tifffile.imread(frames[(channels[0], times[0])]).shape
    dtype = tifffile.imread(frames[(channels[0], times[0])]).dtype

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        import shutil
        shutil.rmtree(tmp_path)
    root = zarr.open_group(str(tmp_path), mode="w")

    shapes, arrays = [], []
    for lvl in range(levels):
        f = 2 ** lvl
        shp = (len(times), len(channels), max(1, nz // f), max(1, ny // f), max(1, nx // f))
        shapes.append(shp)
        arrays.append(root.create_dataset(
            str(lvl), shape=shp, dtype=dtype,
            chunks=(1, 1, min(64, shp[2]), min(256, shp[3]), min(256, shp[4])),
            dimension_separator="/",
        ))

    for ti, t in enumerate(times):
        for ci, c in enumerate(channels):
            src = frames.get((c, t))
            if src is None:
                continue                       # ragged series: leave zeros
            vol = tifffile.imread(src)
            for lvl in range(levels):
                z, y, x = shapes[lvl][2:]
                arrays[lvl][ti, ci, :z, :y, :x] = vol[:z, :y, :x]
                if lvl + 1 < levels:
                    vol = _downsample(vol)

    labels = channel_labels or [f"C{c}" for c in channels]
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": out_path.name.removesuffix(".ome.zarr"),
        "axes": [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {"path": str(lvl), "coordinateTransformations": [
                {"type": "scale", "scale": [time_interval_s, 1.0,
                                            voxel_um * 2 ** lvl,
                                            voxel_um * 2 ** lvl,
                                            voxel_um * 2 ** lvl]}]}
            for lvl in range(levels)
        ],
    }]
    root.attrs["omero"] = {
        "name": out_path.name,
        "channels": [
            {"label": labels[i] if i < len(labels) else f"C{c}",
             "color": _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)],
             "active": True,
             "window": {"start": 0, "end": 300, "min": 0, "max": 65535}}
            for i, c in enumerate(channels)
        ],
    }

    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)
    return out_path


def write_chimerax_script(
    dsr_dir: Path,
    out_path: Path,
    *,
    voxel_um: float = DSR_VOXEL_UM,
    single_names: list[str] | None = None,
) -> Path:
    """A one-click ChimeraX opener for the frame series.

    The stamped OME-TIFFs already carry the voxel size, so this only has to
    group them into a volume series -- ChimeraX treats a glob of separate
    files as separate models otherwise. A single timepoint is just one file
    per channel, opened directly.
    """
    frames = parse_dsr_frames(Path(dsr_dir), single_names)
    channels = sorted({c for c, _ in frames})
    times = {t for _, t in frames}
    lines = [
        "# Generated by opym backfill. Voxel size travels in the OME-TIFF headers;",
        f"# it is {voxel_um} um isotropic, set by the DSR resampling.",
    ]
    for c in channels:
        if len(times) == 1 and not _FRAME_RE.match(frames[(c, min(times))].name):
            lines.append(f'open "{frames[(c, min(times))].resolve()}"')
            continue
        lines.append(f'open "{Path(dsr_dir).resolve()}/*_C{c}_T*.tif" format imagestack vseries true')
    lines.append("volume all style surface")
    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def export_for_viewers(
    dsr_dir: Path,
    out_dir: Path,
    *,
    name: str,
    channel_labels: list[str] | None = None,
    voxel_um: float = DSR_VOXEL_UM,
    single_names: list[str] | None = None,
    output_format: str = "both",
) -> dict:
    """Keep the DSR result in `output_format` (see OUTPUT_FORMATS).

    "tiff" stamps the frames as OME-TIFF (ChimeraX), "both" also builds the
    OME-Zarr (napari), and "ome-zarr" builds the OME-Zarr and then removes the
    DSR TIFF frames -- each only after its zarr copy reads back identical, so
    a failed or partial export never costs the only copy of the result.
    PetaKit5D's MIPs are left alone in every mode.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}")
    dsr_dir, out_dir = Path(dsr_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = parse_dsr_frames(dsr_dir, single_names)
    labels = channel_labels or []

    stamped = 0
    if output_format in ("tiff", "both"):
        for (c, _t), p in sorted(frames.items()):
            label = labels[c] if c < len(labels) else None
            if stamp_ome_tiff(p, voxel_um=voxel_um, channel=label):
                stamped += 1

    summary: dict = {"frames": len(frames), "stamped": stamped,
                     "output_format": output_format, "voxel_um": voxel_um}
    if output_format in ("ome-zarr", "both"):
        zarr_path = write_ome_zarr(dsr_dir, out_dir / f"{name}_dsr.ome.zarr",
                                   voxel_um=voxel_um, channel_labels=channel_labels,
                                   single_names=single_names)
        summary["ome_zarr"] = str(zarr_path)
    if output_format in ("tiff", "both"):
        cxc_path = write_chimerax_script(dsr_dir, out_dir / f"{name}_dsr.cxc",
                                         voxel_um=voxel_um, single_names=single_names)
        summary["chimerax"] = str(cxc_path)
    if output_format == "ome-zarr":
        summary["removed_tiffs"] = remove_frames_verified(frames, Path(summary["ome_zarr"]))

    (out_dir / "viewer_export.json").write_text(json.dumps(summary, indent=1))
    return summary


def remove_frames_verified(frames: dict[tuple[int, int], Path], zarr_path: Path) -> int:
    """Delete each DSR frame only once level 0 of `zarr_path` holds it exactly.

    Raises on the first mismatch, leaving that frame and every later one on
    disk. Returns how many frames were removed.
    """
    import zarr

    level0 = zarr.open_group(str(zarr_path), mode="r")["0"]
    channels = sorted({c for c, _ in frames})
    times = sorted({t for _, t in frames})
    removed = 0
    for (c, t), p in sorted(frames.items()):
        vol = tifffile.imread(p)
        stored = level0[times.index(t), channels.index(c)]
        if stored.shape != vol.shape or not np.array_equal(stored, vol):
            raise RuntimeError(f"OME-Zarr copy of {p.name} does not match -- keeping the TIFFs")
        p.unlink()
        removed += 1
    return removed
