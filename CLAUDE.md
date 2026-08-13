# CLAUDE.md — Bioimaging / opym Orchestration Guide

## Orchestration Rule

**Before starting any non-trivial task, write a step-by-step execution plan to `.claude/current_plan.md`.** Update it as work progresses. This file is the single source of truth for what is being done and why.

---

## Project Layout

Two repos share one `uv` virtual environment. `opym_local` is installed as an editable dependency of `bioimaging`.

```
~/projects/
├── bioimaging/                    # App repo (this repo)
│   ├── run_pipeline_cli.py        # Entry: opym — multiprocess GPFS→GPU pipeline
│   ├── run_napari_opym.py         # Entry: naparym — Napari GUI front-end
│   ├── pyproject.toml             # uv/hatchling; opym editable from ../opym_local
│   ├── JustFile                   # DCV session helpers
│   ├── psf_tools/                 # PSF measurement & channel/fluorophore detection
│   ├── data/                      # PSF TIFs, bead coords, metrics (not in pipeline)
│   ├── scripts/                   # Profiling, MIP export, monitoring (non-production)
│   ├── scratch/                   # Throwaway scripts (never refactor into library)
│   └── tests/                     # pytest suite
│
└── opym_local/                    # Library repo (src layout, editable install)
    └── src/opym/
        ├── core.py                # process_dataset(): OME-TIF→zarr→crop→TIFF/Zarr
        ├── cli.py                 # argparse CLI (entry point: opym)
        ├── petakit.py             # JSON job-ticket writer → MATLAB queue
        ├── local_gpu_worker.py    # Watchdog: polls queue, launches MATLAB server
        ├── batch.py               # Batch orchestration for Jupyter/ipywidgets
        ├── metadata.py            # Processing-log creation
        ├── roi_utils.py           # ROI slice serialization
        ├── utils.py               # OutputFormat enum, path helpers
        ├── submit_opm.py          # High-level job submission
        ├── viewer/                # ndv/napari viewers
        ├── widgets/               # ipywidgets (extractor, averager, decon_viewer)
        └── patches/ + *.m         # MATLAB patches & run_gpu_pipeline.m / server
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| Package manager | `uv` (lockfile: `uv.lock`) |
| Build backend | `hatchling` |
| GPU compute | PyTorch + CUDA 12.6, PetaKit5D (MATLAB engine 24.2) |
| Image I/O | `tifffile` → virtual Zarr (no full-file reads), `zarr<3` |
| GUI | `napari[all]`, `magicgui`, `ipywidgets`, `ndv` |
| Linting | `ruff` (opym_local); enforced via pre-commit |

---

## Data Pipeline (End-to-End)

```
GPFS OME-TIF
  └─ tifffile.aszarr() → virtual Zarr (IFD parse only, zero data I/O)
       └─ ProcessPoolExecutor (≤8 workers, chunked by timepoint)
            └─ each worker: crops ROI → zarr.save → /dev/shm/opym_jobs/<name>.zarr
                 └─ petakit.submit_pipeline_job() writes JSON ticket
                      └─ /dev/shm/petakit_jobs/queue/<PIPELINE_...>.json
                           └─ local_gpu_worker watchdog → MATLAB server
                                └─ PetaKit5D: Decon → DSR (deskew+rotate) → Z-trim
                                     └─ <data_dir>/Decon/<name>.zarr  (final output)
```

Key paths at runtime:
- **Input**: `/mmfs2/scratch/…/<acq_dir>/cell_MMStack_Pos0.ome.tif` (GPFS)
- **Staging RAM disk**: `/dev/shm/opym_jobs/` (zarr per frame, deleted after job)
- **Job queue**: `/dev/shm/petakit_jobs/queue/` → `completed/` / `failed/`
- **Output**: `<data_dir>/Decon/`
- **PSF**: `/mmfs2/scratch/…/PSF/<date>_averaged_psf.tif`
- **ROI cache**: `<data_dir>/master_roi.json`

---

## Architectural Boundaries — Do Not Cross

1. **Python ↔ MATLAB boundary is JSON tickets only.** Never call MATLAB directly from Python worker processes; always write a ticket to `/dev/shm/petakit_jobs/queue/` and let the watchdog dispatch.

2. **No shared file handles across process boundaries.** Each `ProcessPoolExecutor` worker must open its own `tifffile` / zarr store. The main process must `del store` before spawning workers.

3. **opym_local must not import from bioimaging.** Dependency is one-way: `bioimaging` → `opym`. `psf_tools` is internal to `bioimaging` only.

4. **GPFS data is read-only in workers.** Write staging data to `/dev/shm/`, write final output to `<data_dir>/Decon/`.

5. **`scratch/` is never production.** Do not refactor scratch scripts into library modules without a deliberate design step.

6. **GUI entry points stay isolated.** `run_napari_opym.py` and `run_pipeline_cli.py` are thin orchestrators; business logic belongs in `opym`.

---

## Entry Points & Commands

```bash
opym <data_dir>            # CLI pipeline (run_pipeline_cli.py → opym.petakit)
naparym                    # Napari GUI (run_napari_opym.py)
opym-serve                 # GPU watchdog (opym.local_gpu_worker)
opym-receive               # Real-time frame-streaming receiver (opym.stream.receiver)

uv run pytest tests/       # Run test suite
just dcv                   # Create DCV remote desktop session
```

`opym-receive` is a second, independent ingress into the same
`/dev/shm/petakit_jobs` ticket queue `opym-serve` already watches — it
listens for frames pushed live from the acquisition workstation instead of
discovering finished files on GPFS, and stages/tickets each one exactly like
the batch path does. See `opym_local/docs/STREAMING_PROTOCOL.md` for the
wire protocol.

---

## Channel & ROI Conventions

- **5D shape**: `(T, C, Z, Y, X)` or `(T, Z, C, Y, X)` — detect by comparing axis 1 vs 2 size.
- **Dual-camera**: each excitation produces 2 channels (camera 0 = bottom, camera 1 = top).
- **Output channel map** per excitation `e`: indices `e*4+0` (Bot C0), `e*4+1` (Top C0), `e*4+2` (Top C1), `e*4+3` (Bot C1).
- **ROI detection**: midpoint timepoint max-projection, gaussian blur peak-find, cached to `master_roi.json`.
- **Z step**: read from `AcqSettings.txt` → `stepSizeUm`; default 0.3 µm.
- **FFT-friendly sizes**: ROI dimensions rounded up with `scipy.fft.next_fast_len`.
