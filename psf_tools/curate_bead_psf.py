"""Hand-curate isolated beads for PSF measurement in napari.

The automated detect + quality-gate path (``extract_bead_psf`` /
``measure_new_psf``) can let a crop through that has a *second* bead sitting
inside the same box. That neighbor contaminates the averaged PSF (extra lobe,
inflated tails, broken symmetry) and no threshold reliably catches it. This tool
makes contamination *visible* and puts the accept/reject decision back in your
hands:

  * every candidate is drawn as a square the exact size of its crop box, so a
    neighbor bead falling inside the square is obvious in the main view;
  * a "Review crops" montage shows the XY and YZ max-projection of each crop and
    runs a pixel-level secondary-peak check (is there a second bright blob in the
    box, even one detection missed?), flagging contaminated / off-edge beads;
  * you click to add missed beads and delete bad ones on the Beads points layer;
  * "Extract & Average" then averages ONLY the beads you kept (no auto-reject),
    reusing the exact centering / registration / apodization from
    ``extract_bead_psf`` so the result is directly comparable to prior PSFs.

The scoring / montage / extraction helpers take plain arrays (no viewer) so they
can be unit-tested and run headless; ``main()`` launches the interactive viewer.
"""

from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import maximum_filter, uniform_filter

from psf_tools.extract_bead_psf import (
    _extract_crop,
    _soft_ellipsoid_mask,
    _windowed_covariance,
    bootstrap_reregister,
    robust_average,
    score_bead_crops,
)

# --- defaults for the 20260707 YG bead scan (cam0 = crisp green camera) --------
DEFAULT_SCAN = Path(
    "/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload/20260707-YG_PSF/"
    "0p1umBeads_3p8E5Beads/bead_PSF/bead_PSF_MMStack_Pos0.ome.tif"
)
DEFAULT_CAMERA = 0
DEFAULT_ROI = (400, 900, 500, 1720)   # y0, y1, x0, x1 : cam0 upper bead strip
DEFAULT_RZ, DEFAULT_RXY = 40, 15      # -> 81 x 31 x 31 crop, matches prior PSFs
DEFAULT_DZ_UM = 0.1
# Hint thresholds. These drive the *advisory* red/amber/green triage only - the
# montage + your clicks are the source of truth. A sheared OPM PSF has a real
# oblique tail, so off-core energy alone is NOT proof of a neighbor; we only flag
# red on things that are unambiguous.
SEC_FRAC_RED = 1.0    # crop center isn't even the brightest voxel -> neighbor
BLOB_WARN = 4         # this many distinct bright blobs -> amber "look closer"
DEFAULT_SAVE = Path(
    "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260707_curated_psf.tif"
)


# --------------------------------------------------------------------------- #
# Headless-safe helpers                                                        #
# --------------------------------------------------------------------------- #
def load_bead_volume(path, camera=DEFAULT_CAMERA, roi=DEFAULT_ROI):
    """Load one camera's ROI sub-volume from an interleaved dual-camera OME-TIFF
    as a 3D (Z, Y, X) float32 array. Only the requested crop is read."""
    y0, y1, x0, x1 = roi
    store = tifffile.imread(str(path), aszarr=True)
    z = zarr.open(store, mode="r")
    if z.ndim == 4:                      # (Z, C, Y, X)
        vol = np.asarray(z[:, camera, y0:y1, x0:x1])
    elif z.ndim == 3:                    # (Z, Y, X) - single camera already
        vol = np.asarray(z[:, y0:y1, x0:x1])
    else:
        raise ValueError(f"Unexpected bead-scan ndim={z.ndim}, shape={z.shape}")
    return vol.astype(np.float32)


def detect_candidates(vol, threshold=300.0, size=(7, 9, 9), rz=DEFAULT_RZ,
                      require_full_axial=True):
    """Find bright local maxima (background-subtracted) as bead candidates.

    Returns an (N, 3) float array of ZYX centers sorted brightest-first. When
    ``require_full_axial`` is set, only beads whose full +/-rz crop fits inside
    the stack are returned (an 81-slice crop in a 100-slice stack is tight)."""
    med = float(np.median(vol))
    v = np.clip(vol - med, 0, None)
    pk = maximum_filter(v, size=size)
    cand = np.argwhere((v == pk) & (v > threshold))
    if require_full_axial:
        cand = cand[(cand[:, 0] >= rz) & (cand[:, 0] < vol.shape[0] - rz)]
    if len(cand) == 0:
        return np.empty((0, 3), float)
    order = np.argsort(-v[tuple(cand.T)])
    return cand[order].astype(float)


def _core_mask(shape, rz, rxy, core_z=0.45, core_xy=0.5):
    """Boolean ellipsoid marking the central bead's core, centered on the crop.
    Everything outside is where a neighbor bead would show up."""
    Z, Y, X = shape
    zc, yc, xc = (Z - 1) / 2.0, (Y - 1) / 2.0, (X - 1) / 2.0
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    return (
        (zz - zc) ** 2 / (rz * core_z) ** 2
        + (yy - yc) ** 2 / (rxy * core_xy) ** 2
        + (xx - xc) ** 2 / (rxy * core_xy) ** 2
    ) <= 1.0


def secondary_peak_fraction(crop, rz, rxy):
    """Brightest off-core blob as a fraction of the central bead's peak.

    A cleanly isolated bead has only shot noise outside its core (~0.05-0.15);
    a neighbor bead in the box pushes this toward 0.5-1.0. Light smoothing keeps
    single hot pixels from false-triggering (a real bead is many voxels wide)."""
    if crop is None:
        return float("nan")
    sub = np.clip(crop.astype(float) - np.percentile(crop, 10), 0, None)
    core = _core_mask(sub.shape, rz, rxy)
    core_peak = float(sub[core].max()) if core.any() else float(sub.max())
    if core_peak <= 0:
        return 0.0
    off = sub.copy()
    off[core] = 0.0
    off = uniform_filter(off, size=2)
    return float(off.max() / core_peak)


def count_blobs(crop, frac=0.30, minsep_z=6, minsep_xy=5):
    """Number of distinct bright blobs (background-subtracted, lightly smoothed
    local maxima above ``frac`` of the peak) in a crop. An isolated OPM bead
    reads ~1-2 (core, plus at most one node on its oblique tail); a dense
    cluster reads much higher. Advisory only - see the module docstring."""
    if crop is None:
        return 0
    sub = np.clip(crop.astype(float) - np.percentile(crop, 10), 0, None)
    sub = uniform_filter(sub, size=2)
    peak = float(sub.max())
    if peak <= 0:
        return 0
    pk = maximum_filter(sub, size=(minsep_z, minsep_xy, minsep_xy))
    return int(((sub == pk) & (sub > frac * peak)).sum())


def isolation_report(vol, points, rz=DEFAULT_RZ, rxy=DEFAULT_RXY):
    """Per-bead advisory triage. Three independent signals are reported:

      * ``nn_radii`` - distance to the nearest *other* candidate in crop-radii
        units (< 1.0 means another detected bead is literally inside this box);
      * ``sec_frac`` - brightest off-core voxel / core peak (> 1.0 means the
        crop center isn't even the dominant peak -> a neighbor is);
      * ``nblobs``   - count of distinct bright blobs in the box.

    ``flag`` is 'red' when a hard, unambiguous problem is present (neighbor
    point in the box, or center not dominant), 'amber' when the blob count is
    elevated ("look closer"), else 'green'. Because a real OPM PSF is sheared,
    off-core energy by itself is deliberately NOT treated as proof of a
    neighbor - the montage and your clicks make the final call."""
    pts = np.asarray(points, float)
    reports = []
    for i, pt in enumerate(pts):
        nn = np.inf
        for j, o in enumerate(pts):
            if j == i:
                continue
            dz = (pt[0] - o[0]) / rz
            dy = (pt[1] - o[1]) / rxy
            dx = (pt[2] - o[2]) / rxy
            nn = min(nn, float(np.sqrt(dz * dz + dy * dy + dx * dx)))
        crop = _extract_crop(vol, pt, rz, rxy)
        fits = crop is not None
        sec = secondary_peak_fraction(crop, rz, rxy) if fits else float("nan")
        nblobs = count_blobs(crop) if fits else 0
        if not fits:
            flag = "orange"
        elif nn < 1.0 or sec > SEC_FRAC_RED:
            flag = "red"
        elif nblobs >= BLOB_WARN:
            flag = "amber"
        else:
            flag = "green"
        reports.append(
            {"index": i, "point": pt, "nn_radii": nn, "sec_frac": sec,
             "nblobs": nblobs, "fits": fits, "flag": flag,
             "isolated": flag in ("green", "amber")}
        )
    return reports


def _tile(crop, rz, rxy):
    """Composite XY-MIP | YZ-MIP tile for one crop, each independently
    contrast-normalized. Height = crop Y extent; the YZ panel is the wide one
    (full axial extent), where an axially-offset neighbor shows up."""
    sub = np.clip(crop.astype(float) - np.percentile(crop, 10), 0, None)
    xy = sub.max(axis=0)          # (Y, X)
    yz = sub.max(axis=2).T        # (Y, Z)
    xy = xy / (xy.max() + 1e-9)
    yz = yz / (yz.max() + 1e-9)
    gap = np.zeros((xy.shape[0], 2))
    return np.concatenate([xy, gap, yz], axis=1)


def build_montage_rgb(vol, points, rz=DEFAULT_RZ, rxy=DEFAULT_RXY,
                      reports=None, ncols=6):
    """Render a review montage (one tile per bead) to an RGB uint8 array.

    Each tile is ``XY-MIP | YZ-MIP``. Frame color: green = no red flag, amber =
    elevated blob count (look closer), red = neighbor bead in the box, orange =
    crop runs off the stack edge. Titles show the bead index, blob count (n) and
    secondary-peak fraction (f). Returned as an array so it drops straight into a
    napari RGB image layer (no interactive matplotlib backend needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    frame = {"green": "#39d353", "amber": "#d29922", "red": "#f85149",
             "orange": "#f0883e"}
    pts = np.asarray(points, float)
    n = len(pts)
    if n == 0:
        return np.zeros((16, 16, 3), np.uint8)
    if reports is None:
        reports = isolation_report(vol, pts, rz, rxy)

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 1.7),
                             facecolor="#0e1116", squeeze=False)
    axes = axes.ravel()
    for k, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0e1116")
        if k >= n:
            ax.axis("off")
            continue
        rep = reports[k]
        color = frame.get(rep["flag"], "#f0883e")
        crop = _extract_crop(vol, pts[k], rz, rxy)
        if crop is None:
            ax.text(0.5, 0.5, f"#{k}\noff-edge", color=color, ha="center",
                    va="center", fontsize=8, transform=ax.transAxes)
        else:
            ax.imshow(_tile(crop, rz, rxy), cmap="magma", vmin=0, vmax=1,
                      aspect="auto")
            ax.set_title(f"#{k} n={rep['nblobs']} f={rep['sec_frac']:.2f}",
                         color=color, fontsize=8, fontfamily="monospace")
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(2.0)
    fig.tight_layout(pad=0.4)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def average_curated(vol, points, rz=DEFAULT_RZ, rxy=DEFAULT_RXY,
                    combine="trimmed_mean", trim_fraction=0.1,
                    bootstrap=True, apodize=True, rotate_90=True,
                    inner_sigma=6.0, outer_sigma=9.0):
    """Average ONLY the given (hand-curated) bead points into a PSF. No quality
    gate is applied - the points *are* the curation. Reuses the exact centering,
    sub-pixel registration and tail apodization from ``extract_bead_psf`` so the
    output is comparable to previously measured PSFs. Returns (psf, scores)."""
    pts = np.asarray(points, float)
    scores = score_bead_crops(vol, pts, rz, rxy, normalize=True)
    crops = [s["final_crop"] for s in scores]
    if not crops:
        raise ValueError("No valid crops from the selected points.")
    if bootstrap:
        crops = bootstrap_reregister(crops)
    psf = robust_average(crops, combine=combine, trim_fraction=trim_fraction)
    if apodize:
        Z, Y, X = psf.shape
        center = (Z / 2 - 0.5, Y / 2 - 0.5, X / 2 - 0.5)
        cov = _windowed_covariance(psf, center, rz, rxy)
        psf = psf * _soft_ellipsoid_mask(psf.shape, center, cov,
                                         inner_sigma=inner_sigma,
                                         outer_sigma=outer_sigma)
    if rotate_90:
        psf = np.rot90(psf, k=1, axes=(1, 2))
    return psf.astype(np.float32), scores


def save_psf(psf, save_path, dz_um=DEFAULT_DZ_UM):
    """Write the PSF as an ImageJ TIFF carrying the z-step in its 'spacing' tag
    (so downstream decon resamples it to each dataset's z-step)."""
    save_path = Path(save_path)
    tifffile.imwrite(str(save_path), psf.astype(np.float32), imagej=True,
                     metadata={"spacing": float(dz_um), "unit": "um"})
    return save_path


# --------------------------------------------------------------------------- #
# napari launcher                                                             #
# --------------------------------------------------------------------------- #
def main(scan=DEFAULT_SCAN, camera=DEFAULT_CAMERA, roi=DEFAULT_ROI,
         rz=DEFAULT_RZ, rxy=DEFAULT_RXY, dz_um=DEFAULT_DZ_UM,
         save_path=DEFAULT_SAVE):
    """Launch the interactive bead-curation viewer."""
    import napari
    from magicgui import magicgui

    scan = Path(scan)
    vol = load_bead_volume(scan, camera, roi)
    print(f"Loaded {scan.name} cam{camera} ROI{roi} -> {vol.shape}")

    viewer = napari.Viewer(title=f"Bead PSF curation - {scan.name}")
    viewer.add_image(
        vol, name="beads", colormap="magma",
        contrast_limits=[float(np.percentile(vol, 1)),
                         float(np.percentile(vol, 99.9))],
    )
    cand = detect_candidates(vol, rz=rz)
    beads = viewer.add_points(
        cand, name="Beads", ndim=3, size=(2 * rxy + 1), symbol="square",
        face_color="transparent", border_color="#39d353", border_width=0.12,
        out_of_slice_display=True,
    )
    print(f"Auto-detected {len(cand)} candidates. Review, then curate by "
          f"clicking (add/select-delete) on the Beads layer.")

    def _recolor(reports):
        """Best-effort per-point border recolor by triage flag."""
        cmap = {"green": [0.22, 0.83, 0.33, 1.0],
                "amber": [0.82, 0.60, 0.13, 1.0],
                "red": [0.97, 0.25, 0.28, 1.0],
                "orange": [0.94, 0.53, 0.24, 1.0]}
        try:
            colors = [cmap.get(r["flag"], cmap["orange"]) for r in reports]
            if colors:
                beads.border_color = np.array(colors)
        except Exception as exc:  # napari API drift shouldn't break the tool
            print(f"(point recolor skipped: {exc})")

    @magicgui(call_button="Detect beads",
              threshold={"min": 0.0, "max": 65535.0, "step": 25.0})
    def redetect(threshold: float = 300.0, full_axial_only: bool = True):
        pts = detect_candidates(vol, threshold=threshold, rz=rz,
                                require_full_axial=full_axial_only)
        beads.data = pts
        beads.border_color = "#39d353"
        print(f"Detected {len(pts)} candidates at threshold {threshold:.0f}.")

    @magicgui(call_button="Review crops")
    def review():
        pts = np.asarray(beads.data, float)
        if len(pts) == 0:
            print("No beads to review.")
            return
        reports = isolation_report(vol, pts, rz, rxy)
        n_green = sum(r["flag"] == "green" for r in reports)
        state = {"green": "clean", "amber": "look-closer", "red": "NEIGHBOR",
                 "orange": "off-edge"}
        print(f"\n{'#':>3} {'z':>4} {'y':>4} {'x':>4} {'nn_r':>6} {'sec_f':>6} "
              f"{'nblob':>6}  flag")
        for r in reports:
            z, y, x = np.round(r["point"]).astype(int)
            print(f"{r['index']:>3} {z:>4} {y:>4} {x:>4} {r['nn_radii']:>6.2f} "
                  f"{r['sec_frac']:>6.2f} {r['nblobs']:>6}  {state[r['flag']]}")
        print(f"-> {n_green}/{len(reports)} clean (green). Red = neighbor in the "
              f"box (delete these); amber = elevated blob count, verify visually "
              f"in the montage before keeping.\n")
        _recolor(reports)
        rgb = build_montage_rgb(vol, pts, rz, rxy, reports=reports)
        if "Crop Review" in viewer.layers:
            viewer.layers["Crop Review"].data = rgb
        else:
            viewer.add_image(rgb, name="Crop Review", rgb=True)

    @magicgui(call_button="Extract & Average PSF",
              combine={"choices": ["trimmed_mean", "median", "mean"]},
              save_path={"mode": "w"})
    def extract(save_path: Path = Path(save_path), dz_um: float = dz_um,
                combine: str = "trimmed_mean", bootstrap: bool = True,
                apodize: bool = True, green_only: bool = False):
        isolated_only = green_only
        pts = np.asarray(beads.data, float)
        if len(pts) == 0:
            print("No beads selected.")
            return
        if isolated_only:
            reports = isolation_report(vol, pts, rz, rxy)
            pts = np.array([r["point"] for r in reports if r["flag"] == "green"])
            print(f"green_only: keeping {len(pts)}/{len(reports)} beads "
                  f"(convenience pre-filter; manual curation is preferred).")
            if len(pts) == 0:
                print("No green beads - curate manually instead.")
                return
        psf, scores = average_curated(vol, pts, rz, rxy, combine=combine,
                                      bootstrap=bootstrap, apodize=apodize)
        print(f"Averaged {len(scores)} beads. Per-bead isolation/SNR:")
        for s in scores:
            print(f"  #{s['index']:>2} iso={s['isolation']:.2f} "
                  f"snr={s['snr']:.1f} sym={s['mirror_symmetry']:.2f}")
        out = save_psf(psf, save_path, dz_um)
        print(f"Saved PSF {psf.shape} -> {out}")
        if "Curated PSF" in viewer.layers:
            viewer.layers["Curated PSF"].data = psf
        else:
            viewer.add_image(psf, name="Curated PSF", colormap="viridis")

    viewer.window.add_dock_widget(redetect, name="1. Detect", area="right")
    viewer.window.add_dock_widget(review, name="2. Review crops", area="right")
    viewer.window.add_dock_widget(extract, name="3. Extract PSF", area="right")

    napari.run()
    return viewer


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Curate isolated beads for a PSF.")
    p.add_argument("--scan", default=str(DEFAULT_SCAN))
    p.add_argument("--camera", type=int, default=DEFAULT_CAMERA)
    p.add_argument("--roi", type=int, nargs=4, default=list(DEFAULT_ROI),
                   metavar=("Y0", "Y1", "X0", "X1"))
    p.add_argument("--rz", type=int, default=DEFAULT_RZ)
    p.add_argument("--rxy", type=int, default=DEFAULT_RXY)
    p.add_argument("--dz", type=float, default=DEFAULT_DZ_UM)
    p.add_argument("--save", default=str(DEFAULT_SAVE))
    a = p.parse_args()
    main(scan=a.scan, camera=a.camera, roi=tuple(a.roi), rz=a.rz, rxy=a.rxy,
         dz_um=a.dz, save_path=a.save)
