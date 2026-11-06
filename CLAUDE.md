# CLAUDE.md — Bioimaging / opym Orchestration Guide

## Orchestration Rule

**Before starting any non-trivial task, write a step-by-step execution plan to `.claude/current_plan.md`.** Update it as work progresses. This file is the single source of truth for what is being done and why.

---

# bioimaging (services layer of the OPM pipeline on Argus)

bioimaging runs the OPM light-sheet pipeline as services: the `backfill/` package (discovery, registry, watch loop,
staging, tickets, MIP movies, viewer export, storage triage), three driver scripts, `psf_tools/` and the unit templates
in `deploy/`. The library is `opym` (`~/projects/science/opym_local`), installed editable from `../opym_local`; its
CLAUDE.md covers the receiver, live lane and GPU supervisor. Read [README.md](README.md) first. Review and plan:
[docs/review/2026-11-review.md](docs/review/2026-11-review.md), [docs/ROADMAP.md](docs/ROADMAP.md).

## Layout (`~/projects/science/bioimaging`)

- `backfill/`: `cli.py` discovery, pool, watch loop, ticket collection; `pipeline.py` staging, tickets, admission,
  outputs; `viewer_export.py`, `mip_movie.py`, `live_index.py`, `log.py`; `storage_triage/` (read-only suggestions,
  rules in `docs/STORAGE_TRIAGE_RULES.md`).
- `run_backfill_cli.py`: the driver; the logic lives in `backfill/` and opym. `run_pipeline_cli.py` (the old one-off pipeline CLI) was retired; the two `psf_tools` sweeps that used its helpers call `opym.roi_detect` and `opym.metadata` directly.
- `psf_tools/`: PSF and decon v2 trial tooling; only `extraction_plan` is on the production path.
- `deploy/`: unit templates and `opym-logrotate.conf`. `tests/` (real tests only), `scripts/` (with `scripts/probes/`, manual probes), `scratch/`,
  `matlab_legacy/`, `data/` are not production.

## Entry points (`pyproject.toml`)

- Ours: `opym-backfill`, `opym-storage-triage`. From opym: `opym-serve`, `opym-receive`, `opym-receive-backfill`,
  `opym-backfill-send`, `opym-stream-watch`, `opym-live-trace`.
- `opym` (and `opym-add-times`) is opym's own CLI; this repo no longer registers a script of that name. The `opym[server]` dependency
  keeps every script name above resolving, so the units' `uv run opym-receive` etc. are unchanged.
- Contracts (display encoding and palette, completion marker and lock names, z-step env rule, registry path) are imported from opym, never
  copied here: `opym.display`, `opym.stream.bfill`, `opym.metadata.zarr_fallback_z_step`, `opym.registry`. Tests use `opym.testing.*` fixtures
  and a `StatusRegistry` subclass for the fake registry. This repo needs `opym[server]` (the base opym is numpy-only).
- `naparym`, the old napari GUI (`run_napari_opym.py`), was retired 2026-11-06. `naparym-live` is a separate repo, deployed from `~/opt/naparym-live`; this repo no longer registers a `naparym-live` script.

## How production runs

- Live units in `~/.config/systemd/user`, run FROM this checkout and its `.venv`: `opym-serve`, `opym-receive`,
  `opym-receive-backfill`, `opym-backfill` (`--watch 120 --decon-psf <20260910 PSF>`), `opym-logrotate.timer`.
  `opym-live-qc` runs CORE's `celldet-live-qc`. Ports: 5555 receiver, 5556 backfill receiver (loopback),
  8420 dashboard, 8421 viewer.
- Env that matters (table in README): `OPYM_BACKFILL_REGISTRY_PATH` (`/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3`),
  `OPYM_DECON_PSF` (set in three places that must agree), `OPYM_ZARR_DEFAULT_Z_STEP=0.5` (PL-04),
  `OPYM_BACKFILL_MAX_INFLIGHT=4`, `PETAKIT_JOBS_DIR=/dev/shm/petakit_jobs`, `OPYM_LIVE_*`.
- Logs: `logs/<unit>.log` (git-ignored), rotated hourly at 50 MB, 4 kept, `copytruncate`. `journalctl --user` is unreadable.
- Pulling this checkout, or opym, is a partial deploy: pull and restart the units together, and only when asked. The three
  `uv run` units re-sync the production `.venv` on every restart (OR-02). `opym-logrotate` reads its config from this checkout.
- Never edit, switch branch in, or run `uv run` / `uv sync` in `~/projects/science/bioimaging`; work in a worktree
  under `~/projects/.worktrees/bioimaging/<topic>`.

## Run and test

Bare `pytest` is safe from a worktree: `pyproject.toml` limits collection to `tests/`, and
`tests/conftest.py` points `OPYM_BACKFILL_REGISTRY_PATH` and `PETAKIT_JOBS_DIR` at a fresh tmp dir before any project
import, redirects `opym.petakit.QUEUE_DIR` (and the default arguments that captured it) to a per-test tmp queue, and makes
any open/mkdir/replace under `/dev/shm/petakit_jobs` raise. `tests/test_test_safety_guards.py` fails if a guard is removed.
Still: never run tests in the production checkout `~/projects/science/bioimaging`, and never `uv run` there (it re-syncs
the production `.venv`). Run from a worktree with the production venv's interpreter:

```
# from a worktree root; never `uv run`, never in ~/projects/science/bioimaging itself
T=$(mktemp -d); export OPYM_BACKFILL_REGISTRY_PATH=$T/registry.sqlite3 PETAKIT_JOBS_DIR=$T/petakit_jobs \
  OPYM_BACKFILL_LOG_SEEN_DIR=$T/log_seen OPYM_OUTPUT_MIRROR_ROOT=$T/mirror OPYM_BACKFILL_DIR=$T/opym_backfill \
  QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=
unset OPYM_DECON_PSF OPYM_ZARR_DEFAULT_Z_STEP OPYM_LIVE_LANE OPYM_LIVE_FORMAT OPYM_LIVE_QC
HOME=$T/home ~/projects/science/bioimaging/.venv/bin/python -m pytest -q -p no:cacheprovider -m "not gpu"
```

241 pass (2026-11-06, against the `opym[server]` lock). To test against a fresh lock without touching the production `.venv`:
`UV_PROJECT_ENVIRONMENT=<scratch>/venv uv sync --frozen` in the worktree, then use `<scratch>/venv/bin/python` above. The env vars cover paths the conftest does not redirect (log, mirror, backfill dir, PSF and lane
settings). Never import a test module from another directory (conftest isolation is skipped).

## Gotchas

- The scratch "tests" that ran at import (one rewrote `/mmfs2/.../benchmark_zarr_*.zarr`, another wrote `/tmp/mock_data`)
  moved to `scripts/probes/` (see its README), plus `scripts/decon_probe.py` and `scratch/mp_probe.py`. They are manual;
  never point pytest at them.
- `backfill.cli.DEFAULT_REGISTRY_PATH` is evaluated at import, so the conftest sets its env var before importing
  `backfill.cli` and refuses to run if the path is not under its tmp dir. `opym.petakit.QUEUE_DIR` ignores
  `PETAKIT_JOBS_DIR`; production code still writes there unless the caller passes `queue_dir=` (open: ask opym to pass
  `lanes.backfill_queue_dir()`), and only the conftest redirect and guard protect tests.
- Ordering risks in `pipeline.py`: old `Decon/` and DSR are deleted before a new ticket is admitted (PL-01); a superseded
  in-flight ticket is not cancelled (PL-02); errors are logged but not recorded (PL-10).
- Storage triage only suggests. The PI deletes; do not delete datasets under `/mmfs1` or `/mmfs2`, and tests and scripts
  must not write there.
- Channel 2 is magenta everywhere (viewers, OME metadata, movies); never red-green.
- Handedness: canonical = rot90 (PR #39). About 44 datasets processed deskew-only 2026-09-24..28 are the old mirrored
  handedness and not re-run (user decides).
- Z step: the TIFF path defaults to 0.3 um (`pipeline.py:648`, PROC-F10); the zarr path refuses to guess unless
  `OPYM_ZARR_DEFAULT_Z_STEP` (unit: 0.5) or `OPYM_ZARR_ALLOW_DEFAULT_Z_STEP` is set.
- Decon is ON in production; parameters live in `opym.decon_config`.

## Sibling coupling

- opym: one-way (opym must not import bioimaging). Contracts: `.opym_complete.json`, `.opym_backfill.lock`,
  `channel_store_timepoints` (ING-F2), decon fingerprint (PROC-F12), `Decon/DSR_decon` paths.
- opym-dashboard reads the registry, `mip_movies/*`, `storage_verdicts.sqlite3`; opym-viewer hardcodes `Decon/DSR_decon`;
  CORE runs the live-qc unit; cell-cinema, chimerax-ngff, naparym-live open `viewer/*_dsr.ome.zarr`.

## Workflow

Branch, test with the safe command, PR into `bscott711/bioimaging` main, merge it yourself, then pull main. Deploy only
when asked. No time or effort estimates in plans or docs.
