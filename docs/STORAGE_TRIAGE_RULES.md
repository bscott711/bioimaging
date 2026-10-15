# Storage triage rules

`opym-storage-triage` produces **suggestions**. It never deletes, moves or edits anything it inspects; the PI makes
every delete. It writes only `storage_triage.sqlite3` (inventory + hash cache), a dated report directory, and
`storage_verdicts.sqlite3` (read by opym-dashboard), all under `/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/`.

**Target state per dataset worth keeping: one copy of the raw acquisition + its final processed output**
(Decon -> DSR if it exists, else DSR), plus the small provenance files the dashboard and viewers read.

```
opym-storage-triage scan            # walk the filesystems (read-only), classify every byte
opym-storage-triage hash            # SHA-256 the raw files of candidate duplicates (resumable; --dry-run sizes it)
opym-storage-triage report          # check finals, apply rules, write <work-dir>/<date>/ and storage_verdicts.sqlite3
opym-storage-triage verify <suggestions.csv>   # run immediately before deleting: re-derives every row from disk
```

## What each byte is

| Class | Meaning | Ever suggested? |
|---|---|---|
| raw | MMStack `*.ome.tif` + `*_metadata.txt` + `AcqSettings.txt`, or the raw `.ome.zarr` store (which lives *beside* the leaf dir) | only as part of a verified duplicate copy or a human-delete dataset |
| final | the registry's `deskew.output_path` (all four layouts), never guessed from folder names | only with its dataset |
| provenance | `mip_movies`, `MIPs`, `decon_qc`, json/txt/mat/cxc | only with its dataset |
| intermediate | `processed_ngff`, cropped TIFF series, `DSR_nodecon` next to a decon final, `Masks`, ... | rule I1a |
| viewer_export | `viewer/*_dsr.ome.zarr` when it is not itself the final | rule I1b (separate tier: napari / ChimeraX / 3-D viewer read it) |
| unknown | anything unrecognised | **never**; a dataset containing any is suggested per known unit, not as a whole directory |

Units under 50 MiB are inventoried but never suggested.

## Dataset rules (first match wins)

| Rule | Verdict | Condition | Incident it guards against |
|---|---|---|---|
| P2 | keep | status running / in progress / reprocessing / pending, or discovered < 14 days ago | deleting what the backfill is still using |
| P1 | keep | labelled keep (any copy); a seglab training cell (`tstar`) or `gate=pass`; a PSF manifest source scan; pinned in `protected.txt` | **0924_C007 and 0928_C005 were labelled *delete* but are 2 of the 27 seglab training cells.** (A delete label that collides with protection goes to review, not silently kept.) |
| D1 | **suggest delete** (tier B, all copies) | label `delete` with source `human`, AND no other copy labelled differently, AND label newer than the dataset's latest deskew/mip_encode, AND the detector does not see a moving cell, AND not a bead/PSF acquisition | see below |
| R1 | **review** | a human delete that failed any D1 check (each failed check is named); labelled interrogate; pipeline signal flag `dud`/`dead`/`corrupt` with no label | see below |
| K0 | keep | everything else | |

D1/R1 checks and why:

- **source = human.** Accept / Accept-all used to write an auto-detector suggestion into the same table as a click.
- **no conflicting copy label.** `20260507_..._CSF/cell_1` is *keep* in DataUpload and *delete* in microscopy: one acquisition, two answers.
- **label newer than the last reprocess.** 56 of 69 labels (37 of 50 deletes) predate the latest deskew; the zarr geometry fix and decon re-enable changed the images. Re-click Delete in the dashboard to re-confirm on the current movie; the next report promotes it.
- **detector.** `yolo_best.pt` fired on 1 of 5 real OPM datasets (2026-09-21) and scored two real keeps as `no_cell_detected` (2026-10-02). So a detector *miss* is **no evidence either way** and never grounds for a delete; a detector hit with motion contradicts a delete.
- **bead / PSF.** Only `20260910_averaged_psf` records its source scans (`*.manifest.json`); the other averaged PSFs on disk do not.

## Duplicate copies (rule DUP1, tier A1)

Copies are found by content, not path: raw files sharing a size (>= 1 MiB) are candidate pairs, kept only when they share >= 50 % of the smaller dataset's raw bytes; then every raw file is hashed (SHA-256, resumable, throttled).
A copy is **redundant** only if every one of its raw files has a byte-identical twin in the copy we keep. Anything unhashed is "pending", never suggested.
The old report grouped by relative path and called 8 *partial* copies identical (one held 62 files / 248 GB against 191 / 757 GB).

Kept copy, in order: protected (seglab references that copy) > verified final output > pipeline status done > larger raw set > lab share (`jacks.local/microscopy`) > `/mmfs1` DataUpload > `/mmfs2` > path. A redundant copy's outputs are suggested too, but only if the kept copy's final output verifies complete; otherwise it is held with a note.

## Intermediates (I1a / I1b, tiers A2 / A3)

Suggested only for K0/P1 datasets whose **final output verifies complete**: per channel, timepoints 0..T-1 contiguous, T equal across channels and equal to the registry's timepoint count, no empty frames, first and last frame of every channel decode (TIFF series); or every array's chunk grid fully present (zarr). A `DSR_decon` that is not the registry's final is never offered.

## Verify before deleting

`suggestions.csv` carries a fingerprint per row (file count, bytes, newest mtime). `opym-storage-triage verify suggestions.csv` rescans the affected datasets and re-applies every rule; it exits non-zero and names each row that changed on disk or is no longer suggested. After deleting a redundant copy, its registry row (`registry.sqlite3`) still exists; retire it before the next backfill pass.
