# Storage triage rules

`opym-storage-triage` produces **suggestions**. It never deletes, moves or edits anything it inspects; the PI makes
every delete. It writes only `storage_triage.sqlite3` (inventory + hash cache), a time-stamped report directory, and
`storage_verdicts.sqlite3` (read by opym-dashboard), all under `/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/`.

**Target state per dataset worth keeping: one copy of the raw acquisition + its final processed output**
(Decon -> DSR if it exists, else DSR), plus the small provenance files the dashboard and viewers read.

```
opym-storage-triage scan            # walk the filesystems (read-only), classify every byte
opym-storage-triage hash            # SHA-256 the raw files of candidate duplicates (resumable; --dry-run sizes it)
opym-storage-triage report          # check finals, apply rules, write <work-dir>/<date>_<time>/ and storage_verdicts.sqlite3
opym-storage-triage verify <suggestions.csv>   # run immediately before deleting: re-derives every row from disk
```

All four run at low priority (`nice +10`, idle I/O class where `ionice` exists, 4 to 6 workers); `--no-throttle` (before
the subcommand) turns that off and `hash --max-mbps N` caps the total read rate. `report` prints how old the inventory is
and **refuses** one older than `--max-scan-age-hours` (default 72) unless you run `scan --refresh` first, pass `--rescan`,
or accept `--allow-stale`, which labels every verdict row `STALE` for the dashboard. Each `report` writes a new directory
named to the second (a clash gets `-2`); an earlier report is never overwritten.

## What each byte is

| Class | Meaning | Ever suggested? |
|---|---|---|
| raw | MMStack `*.ome.tif` + `*_metadata.txt` + `AcqSettings.txt`, or the raw `.ome.zarr` stores of the sample (every channel store; they live *beside* the leaf dir) | only as part of a verified duplicate copy or a human-delete dataset |
| final | the registry's `deskew.output_path` (all four layouts), never guessed from folder names | only with its dataset |
| provenance | `mip_movies`, `MIPs`, `decon_qc`, json/txt/mat/cxc | only with its dataset |
| intermediate | `processed_ngff`, cropped TIFF series, `DSR_nodecon` next to a decon final, `Masks`, ... | rule I1a |
| viewer_export | `viewer/*_dsr.ome.zarr` when it is not itself the final | rule I1b (separate tier; cell-cinema and ChimeraX-NGFF read only this zarr, and the dashboard's TIFF fallback shows one timepoint and one channel). Offered only when the final holds `*_C#_T#.tif` frames for every channel of the zarr; never when the final is itself a zarr |
| unknown | anything unrecognised | **never**; a dataset containing any is suggested per known unit, not as a whole directory |

Units under 50 MiB are inventoried but never suggested. The floor applies to the per-unit tiers (A2, A3) only; whole-copy rows (A1 duplicate copies, B human deletes) are suggested at any size.

## Dataset rules (first match wins)

| Rule | Verdict | Condition | Incident it guards against |
|---|---|---|---|
| P2 | keep | status running / in progress / reprocessing / pending, or discovered < 14 days ago | deleting what the backfill is still using |
| P1 | keep (also: no A2 / A3 rows) | labelled keep (any copy, including a smaller partial copy that shares a raw file proven identical by hash; equal size alone never links two datasets); a seglab training cell (`tstar`) or `gate=pass`; a PSF manifest source scan; pinned in `protected.txt` | **0924_C007 and 0928_C005 were labelled *delete* but are 2 of the 27 seglab training cells.** (A delete label that collides with protection goes to review, not silently kept.) |
| D1 | **suggest delete** (tier B, all copies) | label `delete` with source `human`, AND no other copy labelled differently, AND label newer than the dataset's latest deskew/mip_encode, AND the detector does not see a moving cell, AND not a bead/PSF acquisition | see below |
| R1 | **review** | a human delete that failed any D1 check (each failed check is named); labelled interrogate; pipeline signal flag `dud`/`dead`/`corrupt` with no label | see below |
| K0 | keep | everything else | |

P1 for any reason withholds A2/A3 (the review asked only for seglab; user may narrow). The verdict reasons and the report note say so.

D1/R1 checks and why:

- **source = human.** Accept / Accept-all used to write an auto-detector suggestion into the same table as a click.
- **no conflicting copy label.** `20260507_..._CSF/cell_1` is *keep* in DataUpload and *delete* in microscopy: one acquisition, two answers.
- **label newer than the last reprocess.** At the first run (2026-10-15) 56 of 69 labels (37 of 50 deletes) predated the latest deskew; the zarr geometry fix and decon re-enable changed the images. Re-click Delete in the dashboard to re-confirm on the current movie; the next report promotes it.
- **detector.** `yolo_best.pt` fired on 1 of 5 real OPM datasets (2026-09-21) and scored two real keeps as `no_cell_detected` (2026-10-02). So a detector *miss* is **no evidence either way** and never grounds for a delete; a detector hit with motion contradicts a delete.
- **bead / PSF.** Only `20260910_averaged_psf` records its source scans (`*.manifest.json`); the other averaged PSFs on disk do not.

## Duplicate copies (rule DUP1, tier A1)

Copies are found by content, not path: raw files sharing a size (>= 1 MiB) are candidate pairs, kept only when they share >= 50 % of the smaller dataset's raw bytes; then every raw file is hashed (SHA-256, resumable, throttled). A pre-cropped zarr sample has one store per channel; every channel store is inventoried and hashed, and a suggested copy lists every one of them.
A copy is **redundant** only if every one of its raw files has a byte-identical twin in the copy we keep. Anything unhashed is "pending", never suggested.
At report time the kept copy's matching raw files are `os.stat`-ed (still present, same size and mtime) and the pair is dropped, with a note, if the two are one storage location: resolved roots that are equal or nested (symlink, bind mount), or any file sharing a `(st_dev, st_ino)` (hardlink farm).
Label checks are wider than the 50 % rule: any dataset sharing even one large raw file with a copy (equal size AND equal SHA-256; an unhashed file links nothing, because equal sizes recur across unrelated datasets) is a sibling, so a smaller partial copy labelled keep still stops a delete.
The old report grouped by relative path and called 8 *partial* copies identical (one held 62 files / 248 GB against 191 / 757 GB).

Kept copy, in order: protected (seglab references that copy) > verified final output > pipeline status done > larger raw set > lab share (`jacks.local/microscopy`) > `/mmfs1` DataUpload > `/mmfs2` > path. A redundant copy's outputs are suggested too, but only if the kept copy's final output verifies complete; otherwise it is held with a note.

## Intermediates (I1a / I1b, tiers A2 / A3)

Suggested only for K0 datasets (never P1) whose **final output verifies complete**: per channel, timepoints 0..T-1 contiguous, T equal across channels and equal to the registry's timepoint count, no empty frames, first, middle and last frame of every channel decode (TIFF series); or every array's chunk grid fully present (zarr). A one-volume final (bead/PSF scans, `*.ome.tif`) must be non-empty and decode. A live-lane zarr is checked against its own `.opym_live.json` (`state: complete`, every (t, c) written, `n_t` equal to the registry's count); zarr writers skip all-zero chunks, so a sparse chunk grid does not mean an incomplete store. Any other zarr must have a full chunk grid or it is held. Cached results are keyed on the check version, so changing a check re-verifies everything. A `DSR_decon` that is not the registry's final is never offered.

A viewer zarr (A3) is offered only if TIFF frames exist for every one of its channels. Its channel count comes from `.opym_live.json`, else from the store's own `omero.channels` (checked against the array's channel axis when readable); if neither gives a positive count the copy is held.

## Verify before deleting

`suggestions.csv` carries a fingerprint per row (file count, bytes, newest mtime) and, for A1 rows, the `kept_copy`. `opym-storage-triage verify suggestions.csv` `os.stat`s the raw files of every kept copy against the inventory the hashes were taken from, rescans the rows' datasets, every kept copy and the rest of their duplicate clusters, and re-applies every rule; it exits non-zero and names each row that changed on disk, is no longer suggested, or whose kept copy is missing, changed, gone from the registry or no longer the kept copy. After deleting a redundant copy, its registry row (`registry.sqlite3`) still exists; retire it before the next backfill pass.
