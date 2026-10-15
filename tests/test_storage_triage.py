"""Storage-triage rules, each pinned to the real incident it guards against
(see docs/STORAGE_TRIAGE_RULES.md). Negative controls: the *_without_* tests
re-run a scenario with one safeguard disabled and require the bad outcome, so
the safeguard is shown to be what prevents it."""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile

from backfill.storage_triage import analyze as az
from backfill.storage_triage import dupes, evidence
from backfill.storage_triage.completeness import verify_tiff_series, verify_zarr
from backfill.storage_triage.inventory import (
    FINAL, INTER, PROV, RAW, UNKNOWN, VIEWER, classify, registry_rows, scan_all,
)
from backfill.storage_triage.rules import Facts, Label, decide
from backfill.storage_triage.store import connect

NOW = datetime(2026, 10, 15, tzinfo=datetime.now().astimezone().tzinfo)
OLD = "2026-08-01T00:00:00+00:00"


def facts(**kw):
    base = dict(key="/d/exp/cell", leaf_dir="/d/exp/cell", status="done", signal_flag="ok",
                discovered_at=OLD, reprocessed_at="2026-09-01T00:00:00+00:00",
                label=Label("delete", "human", "2026-10-01T00:00:00+00:00"), now=NOW)
    base.update(kw)
    return Facts(**base)


# ---------------------------------------------------------------- rules
def test_clean_human_delete_is_suggested():
    assert decide(facts()).rule == "D1"


def test_training_cell_labelled_delete_is_not_suggested():
    """0924_C007 / 0928_C005: labelled delete, but are seglab training cells."""
    v = decide(facts(protected=["seglab training cell (tstar)"]))
    assert v.rule == "R1" and any("protected" in r for r in v.reasons)


def test_without_protection_check_training_cell_would_be_deleted():
    assert decide(facts(protected=[])).rule == "D1"  # negative control


def test_label_conflict_between_copies_goes_to_review():
    """20260507_..._CSF/cell_1: keep on one copy, delete on the other."""
    v = decide(facts(sibling_labels={"/o/cell_1": Label("keep", "human", OLD)}))
    assert v.rule in ("R1", "P1")
    assert v.action != "SUGGEST_DELETE"


def test_label_older_than_reprocess_needs_reconfirmation():
    v = decide(facts(label=Label("delete", "human", "2026-08-10T00:00:00+00:00")))
    assert v.rule == "R1" and any("predates" in r for r in v.reasons)


def test_accepted_suggestion_is_not_a_human_decision():
    v = decide(facts(label=Label("delete", "accepted_suggestion", "2026-10-01T00:00:00+00:00")))
    assert v.rule == "R1"


def test_detector_motion_contradicts_delete():
    auto = {"suggested_label": "keep", "reason": "motion_detected", "max_activity_score": 0.09}
    assert decide(facts(auto=auto)).rule == "R1"


def test_detector_miss_is_not_evidence_either_way():
    """yolo_best.pt fires on ~1/5 real OPM datasets: no_cell proves nothing."""
    auto = {"suggested_label": "delete", "reason": "no_cell_detected"}
    assert decide(facts(label=None, auto=auto)).rule == "K0"
    assert decide(facts(auto=auto)).rule == "D1"  # and does not block a human delete


def test_bead_psf_delete_goes_to_review():
    assert decide(facts(leaf_dir="/d/20260115-YGbeads_in_PDMS/bead_0p1")).rule == "R1"


@pytest.mark.parametrize("status", ["running", "in_progress", "reprocessing", "pending"])
def test_active_datasets_are_untouchable(status):
    assert decide(facts(status=status)).rule == "P2"


def test_new_dataset_inside_settle_window_is_kept():
    assert decide(facts(discovered_at=(NOW - timedelta(days=3)).isoformat())).rule == "P2"


def test_bad_signal_without_label_is_review_not_delete():
    v = decide(facts(label=None, signal_flag="dead"))
    assert v.rule == "R1" and v.action == "REVIEW"


def test_unlabelled_default_is_keep():
    assert decide(facts(label=None)).rule == "K0"


def test_keep_label_protects():
    assert decide(facts(label=Label("keep", "human", OLD))).rule == "P1"


# ---------------------------------------------------------------- classification
@pytest.mark.parametrize("parts,is_dir,final,want", [
    (("cell_MMStack_Pos0.ome.tif",), False, None, RAW),
    (("cell_MMStack_Pos0_3.ome.tif",), False, None, RAW),
    (("cell_MMStack_Pos0_metadata.txt",), False, None, RAW),
    (("AcqSettings.txt",), False, None, RAW),
    (("master_roi.json",), False, None, PROV),
    (("mip_movies", "a.webm"), False, None, PROV),
    (("processed_ngff", "x.zarr", "0", "0.0"), False, None, INTER),
    (("processed_tiff_series_split", "cell_C0_T000.tif"), False, None, INTER),
    (("processed_tiff_series_split", "Decon", "DSR_decon", "cell_C0_T000.tif"), False, None, INTER),
    (("processed_tiff_series_split", "Decon", "MIPs", "cell_C0_T000_MIP_z.tif"), False, None, PROV),
    (("processed_tiff_series_split", "DSR_nodecon", "cell_C0_T000.tif"), False, None, INTER),
    (("viewer", "cell_dsr.ome.zarr", "0", "0"), False, None, VIEWER),
    (("viewer", "viewer_export.json"), False, None, PROV),
    (("somebody_elses_folder", "thing.bin"), False, None, UNKNOWN),
])
def test_classify(parts, is_dir, final, want):
    assert classify(parts, is_dir, final) == want


def test_final_wins_over_everything_below_it():
    final = ("processed_tiff_series_split", "Decon", "DSR_decon")
    assert classify(final + ("cell_C0_T000.tif",), False, final) == FINAL
    assert classify(("processed_tiff_series_split", "DSR_nodecon", "x.tif"), False, final) == INTER


# ---------------------------------------------------------------- completeness
def _tif(p, v=1):
    p.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(p, np.full((2, 4, 4), v, np.uint16))


def _series(d, channels=2, times=3):
    for c in range(channels):
        for t in range(times):
            _tif(d / f"cell_C{c}_T{t:03d}.tif")
    return [(f.name, f.stat().st_size) for f in sorted(d.glob("*.tif"))]


def test_complete_series_verifies(tmp_path):
    ok, detail = verify_tiff_series(_series(tmp_path), tmp_path, 3)
    assert ok, detail


def test_missing_timepoint_fails(tmp_path):
    rows = _series(tmp_path)
    (tmp_path / "cell_C1_T001.tif").unlink()
    rows = [r for r in rows if r[0] != "cell_C1_T001.tif"]
    ok, detail = verify_tiff_series(rows, tmp_path, 3)
    assert not ok and "contiguous" in detail


def test_wrong_timepoint_count_vs_registry_fails(tmp_path):
    ok, detail = verify_tiff_series(_series(tmp_path, times=3), tmp_path, 5)
    assert not ok and "registry expects 5" in detail


def test_truncated_frame_fails(tmp_path):
    rows = _series(tmp_path)
    f = tmp_path / "cell_C0_T002.tif"
    f.write_bytes(f.read_bytes()[:40])
    ok, detail = verify_tiff_series([(n, f.stat().st_size if n == f.name else s) for n, s in rows], tmp_path, 3)
    assert not ok and "unreadable" in detail


def test_zarr_chunk_grid(tmp_path):
    (tmp_path / "0").mkdir()
    (tmp_path / "0/.zarray").write_text(json.dumps({"shape": [4, 4], "chunks": [2, 2]}))
    rows = [("0/.zarray", 10)] + [(f"0/{i}.{j}", 5) for i in range(2) for j in range(2)]
    assert verify_zarr(rows, tmp_path)[0]
    assert not verify_zarr(rows[:-1], tmp_path)[0]


# ---------------------------------------------------------------- duplicates (real hashing)
def _raw(d: Path, name="cell_MMStack_Pos0.ome.tif", size=2 * 2**20, seed=1, extra=()):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(np.random.default_rng(seed).bytes(size))
    (d / "AcqSettings.txt").write_text("x")
    for n, s, sd in extra:
        (d / n).write_bytes(np.random.default_rng(sd).bytes(s))


def _mk(tmp_path, leaves, final_ok=True):
    """Registry + inventory for several leaf dirs, each with raw + a final series."""
    reg = sqlite3.connect(tmp_path / "registry.sqlite3")
    reg.row_factory = sqlite3.Row
    reg.executescript("""
    CREATE TABLE datasets (dataset_key TEXT PRIMARY KEY, root TEXT, leaf_dir TEXT, master_file TEXT,
      has_legacy_decon INTEGER DEFAULT 0, discovered_at TEXT, signal_flag TEXT, expected_timepoints INTEGER,
      actual_timepoints INTEGER, decon_psf TEXT, master_file_fingerprint TEXT, decon_params TEXT);
    CREATE TABLE stage_status (dataset_key TEXT, stage TEXT, status TEXT, started_at TEXT, finished_at TEXT,
      output_path TEXT, ticket_path TEXT, error_message TEXT, consecutive_failures INTEGER DEFAULT 0,
      PRIMARY KEY (dataset_key, stage));""")
    for leaf in leaves:
        final = leaf / "processed_tiff_series_split/Decon/DSR_decon"
        if final_ok:
            _series(final, channels=1, times=2)
        reg.execute("INSERT INTO datasets VALUES (?,?,?,?,0,?,?,?,?,NULL,NULL,NULL)",
                    (str(leaf), str(leaf.parent), str(leaf), str(leaf / "cell_MMStack_Pos0.ome.tif"), OLD, "ok", 2, 2))
        for st in ("deskew", "mip_encode"):
            reg.execute("INSERT INTO stage_status VALUES (?,?,?,?,?,?,NULL,NULL,0)",
                        (str(leaf), st, "done", OLD, "2026-09-01T00:00:00+00:00", str(final) if st == "deskew" else None))
        (leaf / "processed_ngff").mkdir(exist_ok=True)
        (leaf / "processed_ngff/blob.bin").write_bytes(b"0" * 3 * 2**20)
    reg.commit()
    conn = connect(tmp_path / "triage_work.sqlite3")
    scan_all(conn, registry_rows(reg), tmp_path / "mirror", workers=2, log=lambda *_: None)
    return reg, conn


def _run(tmp_path, reg, conn, hash_it=True, monkeypatch=None, **kw):
    if monkeypatch:
        monkeypatch.setattr(az, "MIN_SUGGEST_BYTES", 1 * 2**20)
    rows = {r["dataset_key"]: r for r in registry_rows(reg)}
    if hash_it:
        raw = dupes.load_raw(conn)
        todo = [(f, f.abspath(rows[f.dataset_key]["leaf_dir"], tmp_path / "mirror", rows[f.dataset_key]["master_file"]))
                for f in dupes.files_to_hash(raw, dupes.candidate_pairs(raw))]
        dupes.hash_missing(conn, todo, workers=2, log=lambda *_: None)
    az.check_finals(conn, registry_rows(reg), tmp_path / "mirror", workers=2, log=lambda *_: None)
    return az.analyze(conn, reg, triage_db=tmp_path / "none.sqlite3", auto_db=tmp_path / "none2.sqlite3",
                      seg_db=tmp_path / "none3.sqlite3", psf_dir=tmp_path / "nopsf", pinned_file=tmp_path / "none.txt",
                      mirror_root=tmp_path / "mirror", now=NOW)


@pytest.fixture
def two_copies(tmp_path):
    a = tmp_path / "mmfs1/scratch/jacks.local/microscopy/exp/cell"
    b = tmp_path / "mmfs2/scratch/SDSMT.LOCAL/DataUpload/exp/cell"
    for d in (a, b):
        _raw(d)
    return a, b


def test_identical_copy_is_suggested_and_keeper_is_not(tmp_path, two_copies, monkeypatch):
    a, b = two_copies
    reg, conn = _mk(tmp_path, [a, b])
    A = _run(tmp_path, reg, conn, monkeypatch=monkeypatch)
    dup = [s for s in A.suggestions if s.rule == "DUP1"]
    assert {s.dataset_key for s in dup} == {str(b)}          # mmfs2 copy goes, lab copy stays
    assert all(s.kept_copy == str(a) for s in dup)
    assert not any(s.dataset_key == str(a) and s.tier == "A1" for s in A.suggestions)


def test_same_size_different_bytes_is_not_a_duplicate(tmp_path, two_copies, monkeypatch):
    a, b = two_copies
    _raw(b, seed=2)                                           # same name, same size, other content
    reg, conn = _mk(tmp_path, [a, b])
    A = _run(tmp_path, reg, conn, monkeypatch=monkeypatch)
    assert not [s for s in A.suggestions if s.rule == "DUP1"]


def test_size_only_comparison_would_call_them_duplicates(tmp_path, two_copies):
    """Negative control: what the old path/size report effectively did."""
    a, b = two_copies
    _raw(b, seed=2)
    assert (a / "cell_MMStack_Pos0.ome.tif").stat().st_size == (b / "cell_MMStack_Pos0.ome.tif").stat().st_size


def test_partial_copy_is_a_subset_only_when_every_file_matches(tmp_path, monkeypatch):
    a = tmp_path / "mmfs1/scratch/jacks.local/microscopy/exp/cell"
    b = tmp_path / "mmfs2/scratch/SDSMT.LOCAL/DataUpload/exp/cell"
    _raw(a, extra=[("cell_MMStack_Pos0_1.ome.tif", 2 * 2**20, 7)])
    _raw(b)                                                    # the first file only
    reg, conn = _mk(tmp_path, [a, b])
    A = _run(tmp_path, reg, conn, monkeypatch=monkeypatch)
    assert {s.dataset_key for s in A.suggestions if s.rule == "DUP1"} == {str(b)}
    # but a partial copy holding bytes the full copy lacks is NOT redundant
    _raw(b, name="cell_MMStack_Pos0_9.ome.tif", seed=99)
    reg2 = sqlite3.connect(tmp_path / "registry.sqlite3")
    reg2.row_factory = sqlite3.Row
    conn2 = connect(tmp_path / "triage_work.sqlite3")
    scan_all(conn2, registry_rows(reg2), tmp_path / "mirror", workers=2, refresh=True, log=lambda *_: None)
    A2 = _run(tmp_path, reg2, conn2, monkeypatch=monkeypatch)
    assert not [s for s in A2.suggestions if s.rule == "DUP1"]


def test_unhashed_candidates_are_never_suggested(tmp_path, two_copies, monkeypatch):
    a, b = two_copies
    reg, conn = _mk(tmp_path, [a, b])
    A = _run(tmp_path, reg, conn, hash_it=False, monkeypatch=monkeypatch)
    assert not [s for s in A.suggestions if s.tier == "A1"] and A.pending_files > 0


def test_duplicate_outputs_held_when_no_final_is_verified(tmp_path, two_copies, monkeypatch):
    a, b = two_copies
    reg, conn = _mk(tmp_path, [a, b])
    for d in (a, b):                                           # break both finals
        next((d / "processed_tiff_series_split/Decon/DSR_decon").glob("*_T001.tif")).unlink()
    scan_all(conn, registry_rows(reg), tmp_path / "mirror", workers=2, refresh=True, log=lambda *_: None)
    A = _run(tmp_path, reg, conn, monkeypatch=monkeypatch)
    assert not [s for s in A.suggestions if s.tier == "A1"]
    assert any("not verified complete" in n for n in A.notes)


def test_intermediates_only_suggested_when_final_verified(tmp_path, monkeypatch):
    good = tmp_path / "mmfs1/scratch/SDSMT.LOCAL/DataUpload/exp/good"
    bad = tmp_path / "mmfs1/scratch/SDSMT.LOCAL/DataUpload/exp/bad"
    for d in (good, bad):
        _raw(d, seed=hash(d.name) % 1000)
    reg, conn = _mk(tmp_path, [good, bad])
    next((bad / "processed_tiff_series_split/Decon/DSR_decon").glob("*_T001.tif")).unlink()
    scan_all(conn, registry_rows(reg), tmp_path / "mirror", workers=2, refresh=True, log=lambda *_: None)
    A = _run(tmp_path, reg, conn, monkeypatch=monkeypatch)
    got = {s.dataset_key for s in A.suggestions if s.tier == "A2"}
    assert got == {str(good)}
    assert all("DSR_decon" not in s.path for s in A.suggestions)    # final is never offered


def test_changed_file_changes_fingerprint(tmp_path, monkeypatch):
    d = tmp_path / "mmfs1/scratch/SDSMT.LOCAL/DataUpload/exp/c"
    _raw(d)
    reg, conn = _mk(tmp_path, [d])
    before = {s.path: s.fingerprint for s in _run(tmp_path, reg, conn, monkeypatch=monkeypatch).suggestions}
    (d / "processed_ngff/blob.bin").write_bytes(b"1" * 4 * 2**20)
    os.utime(d / "processed_ngff/blob.bin", (1, 1))
    scan_all(conn, registry_rows(reg), tmp_path / "mirror", workers=2, refresh=True, log=lambda *_: None)
    after = {s.path: s.fingerprint for s in _run(tmp_path, reg, conn, monkeypatch=monkeypatch).suggestions}
    assert before and before != after


def test_pre_source_triage_db_labels_are_human(tmp_path):
    p = tmp_path / "triage.sqlite3"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE triage (dataset_key TEXT PRIMARY KEY, label TEXT NOT NULL, updated_at TEXT NOT NULL)")
    c.execute("INSERT INTO triage VALUES ('k','delete','2026-10-01T00:00:00+00:00')")
    c.commit()
    assert evidence.load_labels(p)["k"].source == "human"


def test_verdict_db_marks_redundant_copy(tmp_path, two_copies, monkeypatch):
    from backfill.storage_triage.report import write_verdicts
    a, b = two_copies
    reg, conn = _mk(tmp_path, [a, b])
    A = _run(tmp_path, reg, conn, monkeypatch=monkeypatch)
    write_verdicts(A, tmp_path / "v.sqlite3")
    got = {r[0]: r[1] for r in sqlite3.connect(tmp_path / "v.sqlite3").execute("SELECT dataset_key, rule FROM verdict")}
    assert got[str(b)] == "DUP1" and got[str(a)] == "K0"
