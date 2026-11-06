# bioimaging roadmap

Ordered from the 2026-11 review ([docs/review/2026-11-review.md](review/2026-11-review.md)); finding ids refer to it.
Scope: S one or two files, low risk; M several files in one subsystem; L several files or repos, or touches live
behaviour. Items 1 and 2 come first because they remove the ways this repo can change production by accident. Items
3 to 9 follow in priority order; items whose files do not overlap can run in parallel. Deploys and restarts happen
only when the user says so. Other-repo ids (BF-, ING-, PROC-) are from the [opym review](../../opym_local/docs/review/2026-11-review.md) and
[opym roadmap](../../opym_local/docs/ROADMAP.md).

| # | Item | Scope | Depends on | Unblocks |
|---|---|---|---|---|
| 1 | Test safety (done 2026-11-06, except the opym `queue_dir` ask) | S | none | safe tests for 3 to 8 |
| 2 | Units | S (deploy needs the user's go) | none | restarts that do not change the venv |
| 3 | Never delete before admission; record errors | M | 1 | 4, 5 |
| 4 | Discovery and readiness | M | 1; opym ING-F2 for PL-08 | 5 |
| 5 | Provenance | M (cross-repo) | 3; user decision | trustworthy `done` |
| 6 | Products | M | 1 | none |
| 7 | Storage triage | M | 1 | none |
| 8 | psf_tools and naparym | S to M | 1 | none |
| 9 | Lint, type debt, dead code | S | none | none |

Item 1 is done: bare `pytest` is safe from a worktree (never in the production checkout, never `uv run` there). The command is in [CLAUDE.md](../CLAUDE.md).

## 1. Test safety
- Status: DONE 2026-11-06 (PR A), except the last bullet.
- What was done: the 15 zero-test scratch files moved out of `tests/` to `scripts/probes/` (README there), with
  `scripts/test_decon.py` and `scratch/test_mp.py` renamed to `*_probe.py`; `pyproject.toml` has `testpaths = ["tests"]`, a
  registered `gpu` marker and `norecursedirs`; `tests/conftest.py` sets `OPYM_BACKFILL_REGISTRY_PATH` and `PETAKIT_JOBS_DIR`
  to a tmp dir before any project import (and refuses to run if `backfill.cli.DEFAULT_REGISTRY_PATH` is not under it),
  redirects `opym.petakit.QUEUE_DIR` and its captured default arguments, and raises on any open/mkdir/replace under
  `/dev/shm/petakit_jobs`; `test_watch_backfill` passes a tmp registry and stubs `start_live_index_thread`;
  `tests/test_test_safety_guards.py` is the negative control.
- Still open: ask opym to pass `queue_dir=lanes.backfill_queue_dir()` (opym BF contract), so production code stops
  depending on the test guard for `QUEUE_DIR`.
- Closes: TI-01 to TI-09, OR-09 (pytest part).
- Unblocks: every later item (they add tests).
- Decision taken: the scratch files are kept under `scripts/probes/`, not deleted.

## 2. Units
- What: run `.venv/bin/<script>` instead of `uv run` in `opym-backfill`, `opym-receive` and `opym-receive-backfill`
  (OR-02, CL-15); fix the `opym-live-qc` template paths to `science/CORE` (OR-01); rename bioimaging's `opym` script
  or drop it (OR-03, CL-08, PSF-16) and fix the stale `sys.path` lines (OR-04); one source for `OPYM_DECON_PSF`
  (OR-06); an instance lock for `opym-backfill` (CL-09); catch a missing PSF at start (CL-15).
- Closes: OR-01, OR-02, OR-03, OR-04, OR-06, CL-08, CL-09, CL-15.
- Note: repo changes are S; installing the units and restarting is a deploy and needs the user's go. After it, a
  dependency change needs an attended `uv sync`.

## 3. Never delete before admission; record errors
- What: clean the old output only after admission and the new ticket succeed (or rename aside, delete later); take
  the admission lock before cleaning; leave a running ticket with mismatched provenance alone, or cancel the queued
  one; write tickets atomically and record `start_stage` before they become visible; record staging and submit
  failures with `finish_stage(failed)` and keep deferrals out of the give-up count; reorder `mip_encode` done against
  reap and export, with its own `try` and an export marker; isolate per-dataset exceptions in `_drain_resolved_tickets`
  and `_run_mip_encode_locked`; make the live-index thread self-healing.
- Closes: PL-01, PL-02, PL-05, PL-10, PL-12, PL-13, PL-11 with CL-06 and PR-03, CL-04, CL-05, PL-15.
- Depends on: 1 (tests for the admission race, a `running` row with a different fingerprint, a failing stage).
- Note: PL-02 and PL-13 touch opym `petakit.py` (ticket write); the per-dataset lock from item 4 can be reused if it
  lands first, but this item does not wait for it.

## 4. Discovery and readiness
- What: settle gate for OME-TIF datasets (CL-03); check the backfill lock before mtime (PL-07); drop or flag a partial
  last volume, and stage by contiguous index (PL-08, with opym ING-F2 `valid_t`); re-run a store that grew since it was
  processed (PL-09); deadlines that start when a task starts, never `failed` for an unstarted task (CL-01); a
  per-dataset lock shared by crop and staging, with a unique tmp name per writer (CL-02, PL-03); a visible deferred
  state with a reason (CL-07); a size-stable check where mtimes may be preserved (PL-14).
- Closes: CL-01, CL-02, CL-03, CL-07, PL-03, PL-07, PL-08, PL-09, PL-14.
- Depends on: 1; opym ING-F2 (opym roadmap item 2) for the `valid_t` form of PL-08.
- Unblocks: 5.
- Decision: whether to cap submissions per pass (CL-01).

## 5. Provenance
- What: record the z step and its source per dataset (PL-04, CL-10); fingerprint the full effective decon parameter
  set plus a PSF content hash and a staging-version constant (PL-06, CL-17; opym roadmap item 7, PROC-F12).
- Closes: PL-04, CL-10, PL-06, CL-17.
- Depends on: 3 (a re-run must not delete before admission); opym `decon_config` and registry columns.
- Decisions for the user: (a) keep `OPYM_ZARR_DEFAULT_Z_STEP=0.5` in the unit or not; (b) re-run the 4 bead datasets
  processed on the default (`NewDay_003`, `Bead_001` x2, `20260901-SVO-YGbeads_camera_test/Bead_001`); (c) whether to
  re-run the 66 grandfathered datasets once the fingerprint is complete. Nothing is deleted without the user.

## 6. Products
- What: temp plus `os.replace` for all webm, jpg and json writes (PR-02, CL-13); colour by channel identity, channel 2
  magenta, no red in the cycle (PR-07, CL-18); align composites and the zarr time axis by T, not rank (PR-04, PR-05,
  CL-11); no deletes on read paths (PR-06); an export ledger and a resumable, never-`rmtree` TIFF removal (PR-01);
  smaller items PR-08 to PR-12, PR-14.
- Closes: PR-01 to PR-12, PR-14, CL-11, CL-13, CL-18.
- Depends on: 1; PR-03 is in item 3.
- Unblocks: none. The dashboard and viewers read these files (PR-02 is visible there).

## 7. Storage triage
- What: `verify` re-checks the kept copy and uses `os.stat` on keeper raw files (ST-01); inode and realpath guard
  (ST-03); inventory every zarr channel store (ST-02); never offer class VIEWER unless TIFFs exist for every channel
  (ST-04); scan-age guard and an interior-frame decode (ST-05, ST-06); P1 protection, wording, time-stamped report
  dirs, a throttle (ST-07 to ST-12); tests for `cmd_verify`.
- Closes: ST-01 to ST-12.
- Depends on: 1. Do ST-01 first; it is the only P1.
- Note: the tool never deletes; the PI does. Fix the suggestions before the next delete round.

## 8. psf_tools and naparym
- What: fix or retire `naparym` (PSF-01, PSF-02); anchor the fluorophore regexes and key the extraction-plan cache by
  mtime (PSF-03, PSF-04); deprecate `phase_retrieve_psf` (PSF-05); mark one-off sweeps as non-library and route them
  through lane admission (PSF-08, PSF-13); crop origin with staged volumes (PSF-06); PSF z step from the bead stack
  (PSF-07); smaller items PSF-09 to PSF-15.
- Closes: PSF-01 to PSF-15.
- Depends on: 1. PSF-03 and PSF-04 touch production extraction, so compare the plan before and after over the existing
  DataUpload folder names.
- Decision for the user: fix or retire `naparym` (last edited 2026-08-04; no unit runs it).

## 9. Lint, type debt, dead code
- What: ruff 224 errors (121 fixable) and 67 files unformatted, mypy 12 errors in 6 files (T07 counts, all at
  `01db054`); remove dead code (PL-16, PL-17, PR-13); delete or archive `scratch/`, `matlab_legacy/`, `.vscode/`;
  boolean env parsing (OR-07); declare directly imported dependencies and a real description (OR-09, D-32);
  docstring fixes (CL-14, D-30); catch-up of `CHANGELOG.md` (D-31) and the 0.4.0 question; re-date
  `docs/STORAGE_TRIAGE_HANDOFF.md` (D-33); `scripts/` housekeeping (D-34, D-35, OR-05, OR-10).
- Closes: OR-05, OR-07, OR-09, OR-10, PL-16, PL-17, CL-12, CL-14, CL-16, PR-13, TI-09.
- Note: add a ruff config first so the counts mean something; confirm that none of the counts predates recent work.
