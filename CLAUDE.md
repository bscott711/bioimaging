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
- `run_backfill_cli.py`, `run_pipeline_cli.py`, `run_napari_opym.py`: drivers; the logic lives in `backfill/` and opym.
- `psf_tools/`: PSF and decon v2 trial tooling; only `extraction_plan` is on the production path.
- `deploy/`: unit templates and `opym-logrotate.conf`. `tests/` (15 real files), `scripts/`, `scratch/`,
  `matlab_legacy/`, `data/` are not production.

## Entry points (`pyproject.toml`)

- Ours: `opym-backfill`, `opym-storage-triage`. From opym: `opym-serve`, `opym-receive`, `opym-receive-backfill`,
  `opym-backfill-send`, `opym-stream-watch`, `naparym-live`, `opym-live-trace`.
- `opym` is SHADOWED: opym registers it too and `.venv/bin/opym` is opym's own CLI. Run `python run_pipeline_cli.py <dir>`.
- `naparym` is BROKEN (PSF-01, PSF-02): it cannot finish a run. Do not rely on it.

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

## Run and test (use only this command)

```
# from this repo's root or a worktree root ONLY; never `uv run`, never a bare `pytest`/`pytest tests/`
T=$(mktemp -d); export OPYM_BACKFILL_REGISTRY_PATH=$T/registry.sqlite3 PETAKIT_JOBS_DIR=$T/petakit_jobs \
  OPYM_BACKFILL_LOG_SEEN_DIR=$T/log_seen OPYM_OUTPUT_MIRROR_ROOT=$T/mirror OPYM_BACKFILL_DIR=$T/opym_backfill \
  QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=
unset OPYM_DECON_PSF OPYM_ZARR_DEFAULT_Z_STEP OPYM_LIVE_LANE OPYM_LIVE_FORMAT OPYM_LIVE_QC
HOME=$T/home ~/projects/science/bioimaging/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_backfill_maintenance.py tests/test_bead_psf_quality.py tests/test_decon_haze_bg.py \
  tests/test_decon_staging.py tests/test_decon_v2_cohort.py tests/test_decon_v2_trial.py \
  tests/test_fluorophore_parsing.py tests/test_live_index.py tests/test_mip_movie.py \
  tests/test_phase_retrieve_psf.py tests/test_storage_triage.py tests/test_synthetic_opm_psf.py \
  tests/test_ticket_collection.py tests/test_viewer_export.py tests/test_watch_backfill.py
```

232 pass (2026-11-06). Set the env vars before Python starts: the registry path is read at import.

## Gotchas

- Scratch "tests" in `tests/` run at import: `test_conver_to_zarr.py` removes and rewrites `/mmfs2/.../benchmark_zarr_*.zarr`,
  another writes `/tmp/mock_data`. That is why the command lists files. Never import a test module from another directory
  (conftest isolation is skipped).
- `backfill.cli.DEFAULT_REGISTRY_PATH` is evaluated at import; `opym.petakit.QUEUE_DIR` ignores `PETAKIT_JOBS_DIR`
  (a test that forgets to patch the submit writes into the production queue).
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
