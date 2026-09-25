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

# The OME-Zarr layout (voxel size, pyramid, chunking, NGFF attrs) lives in
# opym.ome_zarr_writer, shared with the live lane, which writes the same store
# timepoint by timepoint during acquisition; see export_for_viewers.
from opym import ome_zarr_writer  # noqa: E402
from opym.ome_zarr_writer import DSR_VOXEL_UM, PYRAMID_LEVELS  # noqa: E402,F401

_FRAME_RE = re.compile(r"^(?P<prefix>.+)_C(?P<c>\d+)_T(?P<t>\d+)\.tif$")



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
    """2x block mean on each axis; see opym.ome_zarr_writer.downsample2."""
    return ome_zarr_writer.downsample2(vol)


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
    labels = channel_labels or [f"C{c}" for c in channels]
    ome_zarr_writer.create_store(
        tmp_path, n_t=len(times), n_c=len(channels), shape_zyx=(nz, ny, nx), dtype=dtype,
        channel_labels=labels, voxel_um=voxel_um, levels=levels, time_interval_s=time_interval_s,
    )
    for ti, t in enumerate(times):
        for ci, c in enumerate(channels):
            src = frames.get((c, t))
            if src is None:
                continue                       # ragged series: leave zeros
            ome_zarr_writer.write_timepoint(tmp_path, ti, ci, tifffile.imread(src))
    ome_zarr_writer.write_progress(
        tmp_path, n_t=len(times), n_c=len(channels),
        done=[[times.index(t), channels.index(c)] for c, t in frames], state="complete",
    )
    # The store is named after its final path, not the .tmp one it was built at.
    root = zarr.open_group(str(tmp_path), mode="r+")
    ms = root.attrs["multiscales"]
    ms[0]["name"] = out_path.name.removesuffix(".ome.zarr")
    root.attrs["multiscales"] = ms
    omero = root.attrs["omero"]
    omero["name"] = out_path.name
    root.attrs["omero"] = omero

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
    zarr_out = out_dir / f"{name}_dsr.ome.zarr"
    # Decided before stamping: stamping rewrites each frame's metadata (never
    # its pixels), which would otherwise make every frame look newer than the
    # store and force a needless rebuild.
    reuse_store = store_is_current(zarr_out, frames)

    stamped = 0
    if output_format in ("tiff", "both"):
        for (c, _t), p in sorted(frames.items()):
            label = labels[c] if c < len(labels) else None
            if stamp_ome_tiff(p, voxel_um=voxel_um, channel=label):
                stamped += 1

    summary: dict = {"frames": len(frames), "stamped": stamped,
                     "output_format": output_format, "voxel_um": voxel_um}
    if output_format in ("ome-zarr", "both"):
        if reuse_store:
            summary["ome_zarr"] = str(zarr_out)
            summary["ome_zarr_reused"] = True   # e.g. built live, during acquisition
        else:
            zarr_path = write_ome_zarr(dsr_dir, zarr_out,
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


def store_is_current(zarr_path: Path, frames: dict[tuple[int, int], Path]) -> bool:
    """Whether an existing OME-Zarr already holds exactly these DSR frames, so
    it needn't be rebuilt: marked complete, one (t, c) slot per frame and
    every frame written, the same full-resolution shape, and no frame newer
    than the store's last progress write (a reprocessed dataset's new frames
    always force a rebuild). The live lane builds such a store during
    acquisition; so does a previous run of this export."""
    import zarr

    progress = ome_zarr_writer.read_progress(zarr_path)
    if not progress or progress.get("state") != "complete" or not frames:
        return False
    channels = sorted({c for c, _ in frames})
    times = sorted({t for _, t in frames})
    if progress.get("n_t") != len(times) or progress.get("n_c") != len(channels):
        return False
    want = sorted([times.index(t), channels.index(c)] for c, t in frames)
    if progress.get("done") != want:
        return False
    try:
        level0 = zarr.open_group(str(zarr_path), mode="r")["0"]
        first = frames[(channels[0], times[0])]
        with tifffile.TiffFile(first) as tf:
            frame_shape = tuple(tf.series[0].shape)
    except (OSError, KeyError, ValueError):
        return False
    if tuple(level0.shape[2:]) != frame_shape:
        return False
    written = float(progress.get("updated_at", 0))
    return all(p.stat().st_mtime <= written for p in frames.values())


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
