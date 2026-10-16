# Storage triage: handoff (2026-10-15)

Lab notebook (update in place, same URL): https://claude.ai/artifact/3JJMkJgY8x9LES9u97b28m
Regenerate: `scripts/storage_triage_notebook/README.md`. Rules and their incidents: `docs/STORAGE_TRIAGE_RULES.md`.

## State
- Tool `opym-storage-triage` (`scan`, `hash`, `report`, `verify`) is merged (bioimaging #17, #18, #19). Work dir
  `/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/storage_triage/` holds `storage_triage.sqlite3` (inventory, hashes,
  final-output checks) and one dated report directory (`suggestions.csv`, `paths_tier*.txt`, `review_queue.csv`,
  `notes.txt`, `summary.json`). `storage_verdicts.sqlite3` next to `registry.sqlite3` feeds the dashboard badge.
- First full run (2026-10-15, hashing complete, 27.9 TB read): A1 duplicate copies 5.73 TB (100 datasets), A2
  intermediates 13.3 TB, A3 viewer zarr copies 8.0 TB, B your delete labels 9.9 GB, review queue 8.07 TB (77 datasets).
  Raw acquisitions alone are about 68.6 TB after dedup: above one 60 TB bucket.
- Audited independently: 5 random duplicate rows `cmp`-identical, the 8 formerly "identical" partial groups not
  suggested, Tier B labels human and fresh, sampled A2 finals read directly.
- Deployed 2026-10-15 21:00: the dashboard was restarted on opym-dashboard #7 (storage badges + filter, label source,
  no-cell never accepted; verified on /mip: 4 delete-suggested, 77 needs-review, 220 protected, 100 redundant copies).
  CORE #6 (`no_cell`) is merged; the CORE checkout is on `feat/seglab`, and the autotriage tool is run on demand.
- The backfill service was restarted on the staging fix (bioimaging #20) at 2026-10-15 20:24. A second restart (to load
  the SSH-backfill completion-marker check, #24) is queued for when it is idle.

## Blank-output incident (found during this work)
`microscopy/Cell_001..Cell_011` decon/DSR output was all zeros. Cause: the backfill staged zarr stores while Globus was
still copying them; unlanded chunks read as `fill_value: 0`; the eroded mask from the first (zero) frame was empty;
decon multiplied every frame to zero; every stage recorded done. Fixes: bioimaging #19 (blank finals never verify),
#20 (15 min idle before staging, `OPYM_ZARR_SETTLE_SECONDS`; stale staged frames rebuilt). Cell_002..Cell_011 had
their mirror `decon_stage`/`viewer`/`mip_movies` cleared and deskew/mip_encode reset; the backfill reprocesses them.
Cell_001 left alone (GFP 1 timepoint, mScarlet 0). Open: 6 TIFF datasets also have empty masks
(20251214_AR_FLM_HL60/S2, 20260316_dHL60s_D1357/Day5_dHL60s_Attempt_4, 20260506/cell_3, 20260512/cell and cell_2,
20260513/cell), cause unknown (not Globus). 30 datasets have a blank LAST timepoint only (held, may be benign).

## Open items
- PI review queue (`review_queue.csv`): 35 stale delete labels (re-click Delete in the dashboard to re-confirm), 39
  dud/dead/corrupt flags, 3 delete-vs-protected, 6 bead/PSF.
- Retire registry rows of any copy you delete (the registry still lists them).
- SSH backfill (replaces Globus for OME-Zarr): opym_local `docs/SSH_BACKFILL.md`; notebook https://claude.ai/artifact/3KrSf6jB3zKo8Kedk1HTPv. Argus service enabled; PC sender not yet run.
