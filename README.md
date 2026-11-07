# bioimaging

GPU-accelerated processing pipeline for light-sheet microscopy acquisitions
(OME-TIFF or Zarr → deskew/rotate/deconvolve via PetaKit5D/MATLAB → OME-TIFF
and OME-Zarr), with a napari-based GUI front end, PSF measurement tools, a
backfill service that processes every acquisition it finds, and a live lane
that processes each timepoint while an acquisition is still running.

See [`CHANGELOG.md`](CHANGELOG.md) for what changed in each release.

## Requirements

- Python 3.12+, [`uv`](https://docs.astral.sh/uv/)
- A sibling clone of `opym_local` at `../opym_local` — `bioimaging`'s
  `pyproject.toml` installs `opym` as an editable path dependency
  (`../opym_local`), so both repos must be checked out next to each other.
  On Argus both live under `~/projects/science/`:
  ```
  projects/science/
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
git clone <bioimaging-repo-url> bioimaging   # must be a sibling of opym_local (on Argus: ~/projects/science/)
cd bioimaging
uv sync
```

## Commands

Installed as console scripts via `uv sync` (see `pyproject.toml`
`[project.scripts]`):

| Command           | What it does                                                                        |
|-------------------|--------------------------------------------------------------------------------------|
| `opym`            | opym's own crop/split CLI (declared by opym, not by this repo; `run_pipeline_cli.py` was retired) |
| `opym-serve`      | GPU supervisor: keeps one PetaKit5D MATLAB server per GPU running over the job queue |
| `opym-receive`    | Streaming ingress for frames pushed live from acquisition; hosts the live lane        |
| `opym-stream-watch` | Acquisition-side client: streams a local `*.ome.zarr` store's new timepoints to `opym-receive` |
| `opym-receive-backfill` | Receiver for the SSH backfill of finished acquisitions from the rig PC (loopback `127.0.0.1:5556`) |
| `opym-backfill-send` | Rig-side sender for that backfill (the rig runs pymmcore-gui's own copy) |
| `opym-backfill`   | Process every acquisition under the data roots (`backfill/`); `--watch SECONDS` keeps running |
| `opym-storage-triage` | Read-only storage suggestions (`scan`, `hash`, `report`, `verify`); see `docs/STORAGE_TRIAGE_RULES.md`. The PI deletes |
| `opym-live-trace` | Per-hop latency report for a live session, last plane acquired to napari screen       |

`opym-serve`, `opym-receive`, `opym-receive-backfill`, `opym-backfill-send`,
`opym-live-trace` and `opym-stream-watch` are implemented in
the `opym` package (`opym.local_gpu_worker`, `opym.stream.receiver`,
`opym.stream.backfill_receiver`, `opym.stream.backfill_client`,
`opym.stream.trace_report`, `opym.stream.client`); opym declares
these scripts itself (also `opym` and `opym-add-times`); this repo only registers `opym-backfill` and
`opym-storage-triage`, and depends on `opym[server]` so the names resolve. This repo imports opym's contracts
(display palette, completion marker, z-step rule, registry path) instead of copying them.

Tests: see [`CLAUDE.md`](CLAUDE.md) ("Run and test"). Bare `pytest` from a
worktree is safe: it collects only `tests/`, and `tests/conftest.py` redirects the
registry and the PetaKit jobs dir to a tmp dir, and its guard refuses
`open`/`mkdir`/`replace` under `/dev/shm/petakit_jobs`. Do not run tests in the production checkout and do not
use `uv run` there (it re-syncs the production `.venv`). The old scratch scripts
that ran at import now live in `scripts/probes/` and are run by hand only.

```bash
just dcv                # start a DCV remote-desktop session (see JustFile)
```

`opym-backfill` options: `--roots`, `--registry-path`, `--workers`,
`--dry-run` (discovery only, no writes), `--discover-only` (register datasets
as pending, do no work), `--poll-interval`, `--mip-fps`, `--decon-psf`,
`--reprocess-legacy-decon`, `--retry-failed`, `--watch`. See
`opym-backfill --help`. Without `--roots` it walks three default roots
(`/mmfs1/.../jacks.local/microscopy`, `/mmfs1/.../bscott/DataUpload`,
`/mmfs2/.../bscott/DataUpload`, listed in `run_backfill_cli.py`) and uses the
default registry `/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3`.

## Backfill

`opym-backfill` walks the data roots and takes each raw acquisition through
triage (signal check, frame-count check, so likely-good data goes first),
crop and channel remap, a deskew/rotate ticket for the GPU servers, MIP movies
and posters, and viewer export. Per-stage status lives in a SQLite registry
(`--registry-path`, or `OPYM_BACKFILL_REGISTRY_PATH`), so every stage is
resumable and a finished dataset costs one lookup per pass.

- **Decon is opt-in.** With `--decon-psf` (or `OPYM_DECON_PSF`) output goes to
  `DSR_decon/`; without it, to `DSR_nodecon/`. Each dataset records the PSF and
  parameter set it was made with, so changing either re-queues it. Once DSR and
  MIPs exist, the staged decon TIFFs and PetaKit5D's per-frame `Decon/` TIFFs
  are removed; the PSF-generation QC files are copied to `decon_qc/` first.
  `OPYM_KEEP_DECON_INTERMEDIATES=1` keeps everything.
- **Watch mode** (`--watch SECONDS`) re-scans forever. A pass does not wait
  for its tickets; the next pass collects every submitted ticket from the
  registry, marks a ticket that vanished from the queue as failed, and counts
  a finished one only if it ran with the current PSF and decon settings.
- **It yields to live acquisitions.** No pass starts while a live lease is
  held, and the deskew submitters defer while the lane is closed.
  `OPYM_BACKFILL_MAX_INFLIGHT` caps how many backfill tickets are queued or
  running at once (unset means no cap).
- **Zarr acquisitions** from the newer acquisition writer are deskewed per
  timepoint. The z step is read from the store, then the acquisition sidecar;
  if neither has it the dataset is deferred rather than deskewed at a guessed
  scale (`OPYM_ZARR_DEFAULT_Z_STEP` names a step for stores that ship without
  one, `OPYM_ZARR_ALLOW_DEFAULT_Z_STEP=1` accepts a 0.3 um guess).
- **One-format results.** If the live lane already wrote the dataset's
  processed OME-Zarr store (see below), the backfill takes it as the finished
  result and builds MIP movies from its Z-MIP series instead of re-deriving
  them from TIFFs.
- **Readiness.** A zarr store is staged only when it has a pixel array and has
  been idle for 15 minutes (`OPYM_ZARR_SETTLE_SECONDS`, default 900), or carries
  `.opym_complete.json` (written by the SSH backfill). OME-TIFF datasets have no
  such gate yet (see the 2026-11 review, CL-03).
- **Live index.** A thread inside the watch loop (`backfill/live_index.py`)
  registers streamed sessions in the registry within seconds, while they are
  still being acquired.
- **Pools** use a `forkserver` context; a stage that fails 3 times in a row is
  skipped until `--retry-failed`.
- **Storage triage** (`opym-storage-triage`, `backfill/storage_triage/`) only
  suggests what could be deleted; the PI deletes. Rules and handoff:
  `docs/STORAGE_TRIAGE_RULES.md`, `docs/STORAGE_TRIAGE_HANDOFF.md` (a
  2026-10-15 snapshot).

## Live lane

With `OPYM_LIVE_LANE=1` and `OPYM_DECON_PSF` set, `opym-receive` hands each
streamed timepoint to the live lane as soon as all of its channels have
landed: decon, then deskew/rotate, on the GPU servers' live queue and ahead of
any backfill work. While it has live work the receiver holds a lease
(`LIVE_LEASE.json` in the job directory); if a live ticket waits, `opym-serve`
can kill a server working on a backfill ticket and requeue that ticket
(`OPYM_LIVE_PREEMPT`, `OPYM_LIVE_PREEMPT_AFTER_S`).

With `OPYM_LIVE_FORMAT=zarr` the lane runs in memory and writes a processed
OME-Zarr store (bioformats2raw layout) instead of decon/DSR/MIP TIFFs.
`naparym-live` follows that store as it grows. Afterwards the backfill takes
over from the lane's `.live_status.json`: a complete run with the same PSF and
parameters is recorded as deskewed, leaving only MIP movies and viewer export;
a running one is left alone; anything else is reprocessed in batch.

With `OPYM_LIVE_QC=1` the receiver also writes raw-projection sidecars for the
`opym-live-qc` service (CORE's `celldet-live-qc`), which judges each timepoint;
the receiver forwards its verdicts to clients that ask for them.

Wire protocol and the timepoint-by-timepoint latency budget are in the `opym`
repo: `docs/STREAMING_PROTOCOL.md` and `docs/live-view-pipeline.md`.

## Services

`deploy/` tracks the systemd user units the pipeline runs as, so the installed
copies can be diffed against the repo (an installed unit once lost
`--decon-psf` that way). `diff deploy/<unit> ~/.config/systemd/user/<unit>` before
copying. To install, `cp deploy/<unit> ~/.config/systemd/user/` then
`systemctl --user daemon-reload`; restart `opym-receive` between acquisitions, not
during one. Each unit's `WorkingDirectory=` and `ExecStart=` show where it expects
this repo to be checked out (on Argus, `~/projects/science/bioimaging`).

`opym-backfill.service` and `opym-receive.service` both take `OPYM_DECON_PSF` from
one `EnvironmentFile=`, `deploy/opym-decon.env`, which is installed next to the
units (`cp deploy/opym-decon.env ~/.config/systemd/user/`). Install it BEFORE the
two units, and pre-flight with `test -f ~/.config/systemd/user/opym-decon.env`:
a missing file stops the unit (and `opym-receive` stops all streaming ingest), it
does not fall back to deskew-only. Change the PSF there and restart both; they
cannot disagree. `opym-backfill` no longer passes `--decon-psf` (the flag
only set this same variable in-process and still works for a manual run).

`opym-logrotate.conf` is not copied: `opym-logrotate.service` renders it from the
checkout at every run, so pulling the checkout changes log rotation at once.

Pulling the production checkout is a partial deploy; pull and restart together.

**Version coupling.** This bioimaging needs `opym_local` >= `b9189d2419c47f0736b188307f4a76f08002cd65` (`b9189d2`; the output-tree names and `leaf_for_store` come from `opym.store`):
production was at `ae1846e`, which lacks `opym[server]`, `opym.display`, `zarr_fallback_z_step`, `registry_path` and the
`opym.stream.bfill` / `opym.stream.outcome` additions. Pull `opym_local` and `bioimaging` in one step, then restart
`opym-receive`, `opym-receive-backfill`, `opym-backfill`, and last `opym-serve` (it has no re-sync of its own). Pulling
either one alone breaks the units.
All four pipeline units (`opym-backfill`, `opym-receive`, `opym-receive-backfill`,
`opym-serve`) run their script from `.venv/bin/` directly. They no longer use
`uv run`, which re-synced the production `.venv` on every (re)start. The
consequence: after a `git pull` that changes `pyproject.toml` or `uv.lock`, the
environment is NOT updated by a restart; run an attended `uv sync` in the
checkout first (with the units stopped), then restart.

| Unit                     | Runs                                            | Role |
|--------------------------|-------------------------------------------------|------|
| `opym-serve.service`     | `opym-serve`                                    | GPU supervisor. Live tickets go first; a hung ticket's server is killed and the ticket requeued (at most twice). |
| `opym-receive.service`   | `opym-receive`                                  | Streaming ingress with RAM-disk staging and the live lane (loopback `127.0.0.1:5555`). |
| `opym-receive-backfill.service` | `opym-receive-backfill`                  | Receiver for the SSH backfill from the rig (loopback `127.0.0.1:5556`, reached through an SSH tunnel); `Nice=15`, idle IO. Enabled and running on Argus. |
| `opym-backfill.service`  | `opym-backfill --watch 120` (PSF from `opym-decon.env`) | Continuous backfill with decon on. Holds a single-instance lock (below). |
| `opym-live-qc.service`   | CORE's `celldet-live-qc` (from CORE's own venv)  | Live per-timepoint QC, CPU only. |
| `opym-logrotate.timer`   | `logrotate` on `opym-logrotate.conf`, hourly    | Rotates the service logs at 50 MB (4 kept, gzipped, `copytruncate`). Enable with `systemctl --user enable --now opym-logrotate.timer`. |

Other services this pipeline feeds, with their own repos and units: opym-dashboard
(port 8420), opym-viewer (8421), naparym-live-host.

Each unit appends its output to `logs/<unit>.log` in the checkout (git-ignored),
because `journalctl --user` is not readable on the host. Every unit sets
`NUMPY_MADVISE_HUGEPAGE=0`: numpy's huge-page madvise on arrays over 4 MB made
first-touch allocations stall in kernel compaction.

`opym-backfill` stops retrying a dataset once one of its stages has failed
3 times in a row (it stays `failed` on the dashboard); after fixing the cause,
run it once with `--retry-failed`.

`opym-receive` and `opym-backfill` must agree on `OPYM_DECON_PSF` and
`OPYM_BACKFILL_REGISTRY_PATH`; both units read the PSF from `opym-decon.env` and
set the registry path to the same value.

Only one `opym-backfill` can run per registry on this host: it takes a non-blocking
`flock` on `.opym-backfill.lock` beside the registry file (`flock` is node-local on
GPFS, which is enough because only micro001 runs the backfill). If the lock itself
cannot be taken (an errno other than "already held"), it prints a WARNING and runs
without the guard rather than exiting. A manual one-shot or
`--retry-failed` while the service runs exits with status 1 and a message; stop the
service first (`systemctl --user stop opym-backfill`). `--dry-run` writes nothing
and is allowed alongside the service.

Environment variables the units and backfill use:

| Variable | Read by | Meaning |
|----------|---------|---------|
| `OPYM_DECON_PSF` | backfill, receiver | PSF for decon. Unset or empty means deskew-only. The units set it from `deploy/opym-decon.env`; `--decon-psf` sets it in-process for a manual `opym-backfill` run. |
| `OPYM_BACKFILL_REGISTRY_PATH` | backfill, receiver | Status registry. The receiver uses it to forget an earlier run whose name a new run reuses. |
| `OPYM_BACKFILL_MAX_INFLIGHT` | backfill | Cap on backfill tickets queued or running. Unset means no cap. |
| `OPYM_DECON_REPROCESS_LEGACY` | backfill | Same as `--reprocess-legacy-decon`. |
| `OPYM_KEEP_DECON_INTERMEDIATES` | backfill | Keep the decon staging and `Decon/` TIFFs. |
| `OPYM_OUTPUT_FORMAT` | backfill | `tiff`, `ome-zarr` or `both` (default) when the raw store does not say. |
| `OPYM_SKIP_VIEWER_EXPORT` | backfill | Skip the OME-TIFF stamping and OME-Zarr export. |
| `OPYM_ZARR_DEFAULT_Z_STEP`, `OPYM_ZARR_ALLOW_DEFAULT_Z_STEP`, `OPYM_ZARR_MAX_TIMEPOINTS` | backfill | z-step fallback for stores missing one (the backfill unit sets 0.5, which turns the refuse-to-guess guard off); cap on timepoints per dataset for test runs. |
| `OPYM_ZARR_SETTLE_SECONDS` | backfill | Idle time a zarr store needs before it is staged. Default 900. |
| `PETAKIT_JOBS_DIR` | backfill, opym | Queue, lease and claim directory. Default `/dev/shm/petakit_jobs`. |
| `OPYM_BACKFILL_LOG_SEEN_DIR` | backfill | Where log-once markers live. Default `<jobs dir>/log_seen`. |
| `OPYM_BACKFILL_DIR` | storage triage | Work dir for `opym-storage-triage`. Default `/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill`. |
| `OPYM_STREAM_CLIENT_LOG_DIR` | receiver | Where rig client logs land (`logs/rig`). |
| `OPYM_SERVE_SERVERS` | opym-serve | GPU server layout for the supervisor (read in opym; see its docs). |
| `OPYM_BACKFILL_RAW_ROOTS`, `OPYM_BACKFILL_BIND`, `OPYM_BACKFILL_SETTLE_SECONDS` | opym-receive-backfill | Allowed raw roots, bind address, and settle time of the SSH backfill receiver. |
| `OPYM_STREAM_STAGE_ROOT` | receiver | RAM-disk staging root; unset writes straight to each session's raw root. |
| `OPYM_LIVE_LANE`, `OPYM_LIVE_FORMAT`, `OPYM_LIVE_QC` | receiver | Enable the live lane, select the one-format (`zarr`) lane, write QC sidecars. |
| `OPYM_SERVE_HANG_MIN`, `OPYM_SERVE_KILL_HUNG` | opym-serve | Minutes without output before a claim is flagged; kill the hung server. |
| `OPYM_LIVE_PREEMPT`, `OPYM_LIVE_PREEMPT_AFTER_S` | opym-serve | Let a waiting live ticket take a server from backfill work, and after how long. |

## Viewing output

Datasets finished by `opym-backfill` get a `viewer/` directory (see
`backfill/viewer_export.py`) holding a pyramidal OME-Zarr,
`<name>_dsr.ome.zarr`, with named channels for napari, and, unless the
dataset's `output_format` is `ome-zarr`, the DSR frames stamped as OME-TIFF
with the real voxel size plus a `<name>_dsr.cxc` script that opens them in
ChimeraX. `naparym-live` opens that OME-Zarr too.

## Viewing older output in ChimeraX (manual step)

There is no `chimerax` command from this repo. The `chimerax-ngff` bundle (its own
repo) opens our OME-Zarr in ChimeraX; use it first. The wrapper below is the older
fallback. To open a bare
`*_processed.zarr` (the crop and channel-remap store, e.g.
`processed_ngff/<name>_processed.zarr`) in ChimeraX:

```bash
python scripts/make_ngff_wrapper.py <name>_processed.zarr
# prints: open in ChimeraX:  open "<name>_ome.zarr"
```

Then run that `open "..."` command inside ChimeraX yourself, after fixing
voxel size if needed (`volume #N voxelSize <x>,<y>,<z>`) — the wrapper's
pixel calibration for X/Y is a placeholder.

## Decon parameter tuning

Production's OMW deconvolution parameters (`DECON_WIENER_ALPHA`,
`DECON_OTF_CUM_THRESH`, `DECON_HANN_WIN_BOUNDS`, `DECON_DAMP_FACTOR`, and
`DECON_EDGE_EROSION` in the `opym` package's `opym.decon_config`, shared by the
backfill and the live lane and re-exported from `backfill/pipeline.py`) were
picked by eye on a genuinely low-SNR cell
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

The same module fixes the deskew/rotate interpolation at linear (cubic took
13.6 s per frame on CPU against 0.07 s, too slow for the live lane); it is not
part of the parameter fingerprint, so switching it does not re-queue datasets.

Changing any of these values changes what every future decon run looks like
across the whole backfill, so treat a change to them the same way this one
was made: compare on real (ideally low-SNR) data in an artifact like these
before merging, not just by reading numbers.

## Repo layout

- `run_backfill_cli.py` — entry point
- `psf_tools/` — PSF measurement, channel/fluorophore detection, decon trial tooling
- `backfill/` — the backfill pipeline: triage, crop, deskew tickets, MIP movies, viewer export
- `backfill/storage_triage/` — read-only storage suggestions
- `scripts/` — profiling, MIP export, Zarr/ChimeraX conversion helpers (non-production), `storage_triage_notebook/`
- `data/` — bead coordinates, PSF json and png (small files; PSF TIFFs are not tracked)
- `docs/` — storage triage rules and handoff; `docs/review/` and `ROADMAP.md` hold the 2026-11 review and plan
- `matlab_legacy/` — legacy standalone MATLAB demo scripts
- `deploy/` — systemd user units: `opym-serve`, `opym-backfill`, `opym-receive`, `opym-receive-backfill`, `opym-live-qc` (live per-timepoint QC, runs from `~/projects/science/CORE`), and the `opym-logrotate` timer
- `CHANGELOG.md` — release notes; `uv.lock` — pinned environment
- `scratch/` — throwaway scripts, never promoted into the library
- `tests/` — pytest suite (see `CLAUDE.md` for the command); `scripts/probes/` holds the manual probes that used to sit here

See `CLAUDE.md` for the services layout, production notes and the safe test command,
and [`docs/review/2026-11-review.md`](docs/review/2026-11-review.md) /
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the current review and plan.
