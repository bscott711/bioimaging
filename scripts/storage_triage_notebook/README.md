# Storage triage lab notebook (artifact generator)

Builds the interactive notebook artifact from the storage-triage report and the databases it reads, so every
number on the page is computed from its source at generation time.

```bash
S=/some/scratch/dir                       # holds the two sizing files below
python3 size_unregistered.py $S           # sizes folders under the data roots that no registered dataset covers -> $S/unregistered_sizes.json
python3 mask_scan.py $S/masks.json        # (bioimaging venv) reads every Decon/Masks/*_eroded.zarr; 0% nonzero = empty mask
python3 collect.py $S <report_dir>        # -> $S/data.json   (report_dir = .../storage_triage/<date>)
python3 build.py $S                       # template + data -> $S/notebook.html
```

Publish `notebook.html` with the Artifact tool, to the SAME artifact URL (see `docs/STORAGE_TRIAGE_HANDOFF.md`).
Static text in `notebook.tmpl.html` that quotes a number not computed in `collect.py` carries its source in the
card ("CORE docs/ARGUS.md", a git commit, ...); re-check those when the underlying fact can change.
