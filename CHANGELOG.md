# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-30

The first release with a changelog. It rolls up everything since 0.2.0
(2026-07-01): the bulk backfill and its always-on watch mode, deconvolution
re-enabled with tuned parameters, the real-time streaming receiver, and the
live lane that processes each timepoint during acquisition. Pull request
numbers are given where a change came in through one. Work from August
onward reached `main` through #4, which brought `main` up to what production
was already running; the early July changes were committed directly.

### Added

- **Bulk backfill** (`opym-backfill`, `backfill/`; #4). Discovers every raw
  acquisition under the configured data roots, triages it (signal check,
  frame-count check, priority ordering so likely-good data goes first), crops
  and converts it, submits a deskew/rotate ticket to the GPU servers, and
  encodes MIP movies and posters (WebM/VP9, green/magenta composites for two
  channels). Progress is recorded per stage in a SQLite status registry, so
  every stage is resumable and skipped when already done.
  - `--watch SECONDS` keeps discovery running as a service; one failed pass is
    logged and retried rather than ending the loop.
  - Also `--dry-run`, `--discover-only`, `--decon-psf` and
    `--reprocess-legacy-decon`.
  - Time-lapse `zarr-precropped` acquisitions from the newer acquisition
    writer are deskewed per timepoint.
- **Deconvolution in the backfill** (#4). Opt-in through `--decon-psf` or
  `OPYM_DECON_PSF`, writing to `DSR_decon/` and leaving existing `DSR_nodecon/`
  output alone. Parameters are the "super4" set picked on low-SNR data
  (wienerAlpha 0.20, OTF cumulative threshold 0.90, Hann window bounds
  [0.4, 1.0], damp factor 2, plus a 3-voxel edge erosion). Each dataset
  records which PSF and parameter set produced it, so changing either
  re-queues the affected datasets instead of silently keeping stale output.
  Once DSR and MIPs exist, the staged decon TIFFs and PetaKit5D's per-frame
  `Decon/` TIFFs are removed, after copying the PSF-generation QC files to
  `decon_qc/`; set `OPYM_KEEP_DECON_INTERMEDIATES=1` to keep everything while
  tuning.
- **Viewer-ready output** (#4). Each finished dataset gets, under `viewer/`,
  stamped OME-TIFF frames and a ChimeraX opener script, and a pyramidal
  OME-Zarr with named channels for napari. `output_format`
  (`tiff` | `ome-zarr` | `both`) is read from the raw store, else from
  `OPYM_OUTPUT_FORMAT`; `ome-zarr` removes each TIFF frame only after its
  zarr copy reads back identical. `OPYM_SKIP_VIEWER_EXPORT` turns it off.
- **Streaming ingress** (`opym-receive`, `opym-stream-watch`; #3, #4). The
  receiver takes frames pushed live from the acquisition workstation, stages
  them on a RAM disk (`OPYM_STREAM_STAGE_ROOT`), and drains them to the
  durable raw store with a byte-for-byte check (#3).
- **Live lane.** Deconvolves and deskews each streamed timepoint as soon as all
  of its channels have landed, ahead of backfill work. The backfill side:
  - Backfill yields to live acquisitions: no pass starts while a live lease is
    held, and both deskew submitters defer while the backfill lane is closed
    (#5).
  - `OPYM_BACKFILL_MAX_INFLIGHT` caps backfill tickets queued or running at
    once (the service uses 4, two per GPU server), so tickets are built
    shortly before they run, with current parameters (#5).
  - Hand-off: the backfill reads the live lane's `.live_status.json`. A
    running lane is left alone; a complete one with matching PSF and
    parameters and every frame on disk is recorded as deskew done, leaving
    only MIP encoding and viewer export; anything else is reprocessed in
    batch (#6). Enabled with `OPYM_LIVE_LANE=1` on the receiver (#6).
  - One-format lane: with `OPYM_LIVE_FORMAT=zarr` the live lane lands a
    processed OME-Zarr store instead of decon/DSR/MIP TIFFs. The backfill
    recognises that store as the finished result and builds MIP movies from
    its Z-MIP series (#11). The repo's `opym-receive.service` now carries this
    setting (#13).
- **`naparym-live`** (#8). Console script for following a live acquisition in
  napari. Viewer export now builds its OME-Zarr through the same writer the
  live lane uses, so the two produce the same layout, and a store that already
  holds exactly a dataset's frames is reused instead of rebuilt.
- **Live QC** (#9). `opym-live-qc.service` runs CORE's `celldet-live-qc` on the
  live session; with `OPYM_LIVE_QC=1` the receiver writes the raw-projection
  sidecars it reads and forwards its verdicts to clients that ask for them.
- **`opym-live-trace`** (#10). Per-hop latency report for the live path, from
  the last plane acquired to the napari screen.
- **Tracked service units** in `deploy/`: `opym-serve` (#4), `opym-backfill`,
  `opym-receive`, `opym-live-qc` (#9). Only installed copies existed for some
  of these before, and they had already drifted once.
- **PSF and decon tooling** (`psf_tools/`, `scripts/`). Automatic bead PSF
  curation and quality scoring, phase-retrieval PSF fitting, an interactive
  bead hand-curation tool, an RL-versus-OMW comparison harness, decon
  parameter-sweep report and render scripts, a duplicate-dataset report, and
  `scripts/make_ngff_wrapper.py` to open Zarr v2 output in ChimeraX.
- **Tests** for ticket collection (#7), viewer export, watch-mode pausing and
  in-flight caps, decon staging and MIP movies. `ome-types` and `xmlschema` are
  now test dependencies for OME-XML schema checks (#11).

### Changed

- **Decon and DSR settings now come from `opym.decon_config`**, the same source
  the live lane uses, so live and batch output cannot drift apart. The
  constants are still re-exported from `backfill/pipeline.py`. DSR
  interpolation is now **linear** instead of cubic, which took 13.6 s per frame
  on CPU against 0.07 s, so the live lane can keep pace with acquisition. The
  decon parameters and their fingerprint are unchanged, so no dataset is
  reprocessed because of it (#6).
- **Pipeline dispatch is batched** by PSF and timepoint, with a larger batch
  limit (30) and single-chunk staging writes for GPU input crops.
- **Watch-mode passes no longer wait on their slowest ticket.** Leftover
  tickets are re-adopted from the registry on the next pass, so one hung
  ticket cannot stop discovery of new data (#4).
- **Services log to files** under `logs/` (ignored by git), since
  `journalctl --user` is not readable on the host (#4, #5, #9, #12).
- **Services opt out of numpy's huge-page madvise**
  (`NUMPY_MADVISE_HUGEPAGE=0`). With transparent-huge-page defrag set to
  `madvise`, first touch of large arrays could stall in direct compaction; on a
  paced replay it cut the p95 from last plane to processed from 0.94 s to
  0.69 s (#13).
- **Hang handling and lane priority** are explicit in `opym-serve.service`:
  `OPYM_SERVE_HANG_MIN`, `OPYM_SERVE_KILL_HUNG`, `OPYM_LIVE_PREEMPT` and
  `OPYM_LIVE_PREEMPT_AFTER_S` (#4, #5).
- Two-channel composite MIPs use green/magenta instead of cyan/magenta (#4).

### Fixed

- **Honest done status** (#7). A dataset was shown as done while its reprocess
  was still in flight. Each pass now first collects every `running` deskew
  ticket from the registry. A ticket in no queue directory is marked failed
  as lost. A finished ticket counts only if it ran with the current PSF and
  decon settings; output made with older settings is re-queued instead of
  grandfathered. Every resubmit resets MIP encoding, so output about to be
  replaced is never encoded. Zarr datasets record declared versus written
  timepoints, so an aborted run reads `2/100`.
- **Streamed run names being reused** (#14). `opym-receive.service` now has the
  backfill's `OPYM_BACKFILL_REGISTRY_PATH`. When a new run reuses a name, the
  receiver moves the earlier run aside and forgets it in the registry;
  without the path the backfill treated the new run as already processed.
- **Stale decon output** is cleaned on any resubmission, not only after a
  failure; a leftover mask crashed MATLAB and, worse, would have reused old
  pixels under the new parameter fingerprint (#4).
- **Decon iteration count** restored to 25; a leftover test value of 10 was
  silently under-deconvolving every production frame.
- **Zarr backfill geometry and completeness** (#4): the z step is read from the
  store, then the acquisition sidecar, and is otherwise refused rather than
  guessed (`OPYM_ZARR_ALLOW_DEFAULT_Z_STEP`, `OPYM_ZARR_DEFAULT_Z_STEP`);
  aborted acquisitions only mirror timepoints every channel finished;
  single-timepoint stores no longer fail with blosc decompression errors;
  per-timepoint mirrors handle stale leftovers; and the DSR path and MIP
  filename lookups for zarr datasets now find their files.
- **Stalled backfill** (#4). One slow or hung dataset in the triage phase no
  longer blocks the whole pass.
- **Unreadable inputs** (#4). Unwritable raw directories are mirrored under
  `OPYM_OUTPUT_MIRROR_ROOT`; a corrupt MMStack sibling is repaired from its
  valid leading prefix; permanently dead files are skipped until they change.
- **Single-timepoint zarr datasets** were silently skipped by viewer export
  (#4).
- **mScarlet fluorophore names** written as bare `Sca`/`Sc` are recognised
  (#4).

## Before 0.3.0

`0.2.0` (2026-07-01, #1) was the end-to-end GPU pipeline: the `opym` CLI and
`naparym` GUI, a unified in-memory GPU decon and deskew/rotate stage fed by
PetaKit5D through JSON tickets and a RAM-disk queue, GPFS-friendly parallel
reads, the `opym-serve` GPU watchdog, and angle-sweep and comparison tooling.
It also carried the fix for a decon/DSR regression: deconvolution ignored the
PSF's z step, and the Python-side crop staging skipped the axis-order
correction the older MATLAB cropper applied. `0.1.0` was the initial project
version, from June 2026. Neither version was tagged or had a changelog.

[Unreleased]: https://github.com/bscott711/bioimaging/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/bscott711/bioimaging/releases/tag/v0.3.0
