# OME-Zarr Consolidation — Move to opym-serve

## Status: COMPLETE

## Task
Move OME-Zarr consolidation out of a polled background watcher process and into
`opym-serve`, which already knows when all MATLAB jobs are done.

## Goal
After `opym <data_dir>` submits MATLAB jobs and exits, `opym-serve` automatically
assembles `Decon/<base>.ome.zarr` once MATLAB finishes — no separate process, no polling.

## Affected Components
- NEW `opym_local/src/opym/consolidate.py`
- `opym_local/src/opym/local_gpu_worker.py`
- `bioimaging/run_pipeline_cli.py`

## Steps
- [x] Create `opym/consolidate.py` with `consolidate_to_ome_zarr()` (zero-copy hardlink)
      and `run_pending_consolidations(base_dir)` (scans tickets for pending sidecars)
- [x] Import and call `run_pending_consolidations` in `local_gpu_worker.py` after both
      MATLAB `p1.wait()` / `p2.wait()` return
- [x] Remove `consolidate_to_ome_zarr`, `_watch_and_consolidate`, `--watch-consolidate`,
      and `subprocess.Popen` watcher from `run_pipeline_cli.py`; import from `opym.consolidate`

## Flow After This Change
1. `opym <data_dir>` writes `Decon/.opym_consolidate.json` sidecar, submits MATLAB jobs, exits
2. `opym-serve` launches two MATLAB servers and blocks on `p1.wait()` / `p2.wait()`
3. Both servers idle out → `run_pending_consolidations(BASE_DIR)` is called
4. Scans `completed/` + `failed/` tickets for unique `dataDir` values
5. Any `dataDir` with `.opym_consolidate.json` gets consolidated (hardlinks, <1s)
6. Sidecar removed; `out.ome.zarr` is immediately drag-droppable in Napari

## Constraints & Risks
- **opym_local must not import from bioimaging** (CLAUDE.md rule 3). Consolidation logic
  lives entirely in `opym/consolidate.py`; `run_pipeline_cli.py` imports from it.
- **Output shape unknown until first MATLAB job finishes.** DSR changes Z/Y dimensions,
  so we can't pre-write `.zarray` shape. Sidecar is still necessary for `base_name`,
  `expected_zarrs`, `xy_pixel_um`, `t_interval_s`, `channel_names`.
- **Hardlinks require same filesystem.** Falls back to symlinks automatically if
  GPFS mount differs from Decon output dir. GPFS does support hardlinks within one fs.
- **opym-serve must be running** for auto-consolidation. Manual fallback: `opym <dir> --consolidate`

## Notes
- `run_pending_consolidations` reads ALL completed/failed tickets, not just ones from this
  session, but the sidecar's existence gates the work — already-consolidated datasets have
  no sidecar and are skipped.
- Consolidation output is tee'd to `Decon/consolidation.log` and to `opym-serve` stdout.
