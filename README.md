# bioimaging

GPU-accelerated processing pipeline for light-sheet microscopy acquisitions
(OME-TIFF → deskew/rotate/deconvolve via PetaKit5D/MATLAB → Zarr), with a
napari-based GUI front end, PSF measurement tools, and a backfill CLI for
reprocessing older acquisitions.

## Requirements

- Python 3.12+, [`uv`](https://docs.astral.sh/uv/)
- A sibling clone of `opym_local` at `../opym_local` — `bioimaging`'s
  `pyproject.toml` installs `opym` as an editable path dependency
  (`../opym_local`), so both repos must be checked out next to each other:
  ```
  projects/
    bioimaging/
    opym_local/
  ```
- CUDA 12.6 GPU + PyTorch (pulled from the `pytorch-cuda` index automatically)
- A licensed MATLAB R2024b install with the MATLAB Engine for Python
  (`matlabengine==24.2.*`) and PetaKit5D, for the decon/DSR stage
- **ChimeraX is not managed by this repo.** If you want to view processed
  volumes in ChimeraX, install it separately as its own application; this
  repo only provides a helper script to make its Zarr output openable there
  (see below).

## Setup

```bash
git clone <opym_local-repo-url> opym_local
git clone <bioimaging-repo-url> bioimaging   # must be a sibling of opym_local
cd bioimaging
uv sync
```

## Commands

Installed as console scripts via `uv sync` (see `pyproject.toml`
`[project.scripts]`):

| Command        | What it does                                                 |
|----------------|---------------------------------------------------------------|
| `naparym`      | napari GUI front end for the pipeline (`run_napari_opym.py`)  |
| `opym <dir>`   | CLI pipeline: GPFS OME-TIFF → GPU decon/DSR → Zarr             |
| `opym-serve`   | GPU watchdog worker that dispatches queued jobs to MATLAB      |
| `opym-receive` | Streaming ingress for frames pushed live from acquisition      |
| `opym-backfill`| Reprocess older acquisitions (`backfill/`)                     |

Also:
```bash
uv run pytest tests/   # test suite
just dcv                # start a DCV remote-desktop session (see JustFile)
```

## Viewing output in ChimeraX (manual step)

There is no `chimerax` command from this repo. To open pipeline output in
ChimeraX:

```bash
python scripts/make_ngff_wrapper.py <name>_processed.zarr
# prints: open in ChimeraX:  open "<name>_ome.zarr"
```

Then run that `open "..."` command inside ChimeraX yourself, after fixing
voxel size if needed (`volume #N voxelSize <x>,<y>,<z>`) — the wrapper's
pixel calibration for X/Y is a placeholder.

## Decon parameter tuning

Production's OMW deconvolution parameters (`DECON_WIENER_ALPHA`,
`DECON_OTF_CUM_THRESH`, `DECON_HANN_WIN_BOUNDS`, `DECON_DAMP_FACTOR` in
`backfill/pipeline.py`) were picked by eye on a genuinely low-SNR cell
(Cell_005, `20260917-SVO-memNG-mScar2xFYVE-FLM-Macropinocytosis`), not left at
PetaKit5D's defaults. The two interactive comparison artifacts below are the
record of that decision — every variant's images, per-plane Z-scrub, and
noise/spike metrics, side by side against the raw (no-decon) data:

- **[Decon Sweep Bench](https://claude.ai/artifact/GxW3QbewXrxxZxgtHLPdiP)** —
  the original 22-variant sweep (α ladder, OTF threshold, Hann window,
  iterations, damp factor, plain-RL reference) that first identified which
  knobs mattered.
- **[Decon Sweep Refinement](https://claude.ai/artifact/4yjCDqqvSugN7t7aGitPdn)** —
  the follow-up sweep combining the first round's picks into single
  "super" variants. **`super4`** (wienerAlpha 0.20, OTFCumThresh 0.90,
  hannWinBounds [0.4, 1.0], dampFactor 2) is the current production default:
  it cut spurious >12σ spike counts to roughly a third of what any single
  changed knob achieved alone, at comparable brightness.

Changing any of these values changes what every future decon run looks like
across the whole backfill, so treat a change to them the same way this one
was made: compare on real (ideally low-SNR) data in an artifact like these
before merging, not just by reading numbers.

## Repo layout

- `run_pipeline_cli.py`, `run_napari_opym.py`, `run_backfill_cli.py` — entry points
- `psf_tools/` — PSF measurement, channel/fluorophore detection
- `backfill/` — reprocessing pipeline for older acquisitions
- `scripts/` — profiling, MIP export, Zarr/ChimeraX conversion helpers (non-production)
- `matlab_legacy/` — legacy standalone MATLAB demo scripts
- `deploy/` — systemd unit for `opym-backfill`
- `scratch/` — throwaway scripts, never promoted into the library
- `tests/` — pytest suite

See `CLAUDE.md` for the detailed data-flow architecture and boundaries.
