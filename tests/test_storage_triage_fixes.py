"""Storage-triage review findings ST-01 to ST-12 (docs/review/2026-11-review.md).

The tool only suggests deletions; each test pins one way a suggestion could have been
unsafe. Tests marked "negative control" re-run the scenario with the safeguard disabled
and require the unsafe outcome, so the safeguard is shown to be what prevents it.
Everything lives in `tmp_path`; nothing touches /mmfs* or the production registry.
"""

import json
import os
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from test_storage_triage import NOW, OLD, _mk, _raw, _series, _tif

from backfill.storage_triage import analyze as az
from backfill.storage_triage import cli, dupes
from backfill.storage_triage.completeness import CHECK_VERSION, verify_tiff_series
from backfill.storage_triage.inventory import registry_rows, scan_all, scan_dataset
from backfill.storage_triage.report import write_report, write_verdicts
from backfill.storage_triage.store import connect

LAB = "mmfs1/scratch/jacks.local/microscopy/exp"      # ranks first: the keeper
SHARE = "mmfs2/scratch/SDSMT.LOCAL/DataUpload/exp"    # the redundant copy


@pytest.fixture(autouse=True)
def small_floor(monkeypatch):
    monkeypatch.setattr(az, "MIN_SUGGEST_BYTES", 1 * 2**20)


def _hash_all(tmp_path, reg, conn):
    rows = {r["dataset_key"]: r for r in registry_rows(reg)}
    raw = dupes.load_raw(conn)
    todo = [(f, f.abspath(rows[f.dataset_key]["leaf_dir"], tmp_path / "mirror", rows[f.dataset_key]["master_file"]))
            for f in dupes.files_to_hash(raw, dupes.candidate_pairs(raw))]
    dupes.hash_missing(conn, todo, workers=2, log=lambda *_: None)


def _hash_every_raw_file(tmp_path, reg, conn):
    """Hash all raw files, not only the candidate-pair ones (as `hash` does for D1-bound datasets)."""
    rows = {r["dataset_key"]: r for r in registry_rows(reg)}
    todo = [(f, f.abspath(rows[f.dataset_key]["leaf_dir"], tmp_path / "mirror", rows[f.dataset_key]["master_file"]))
            for fs in dupes.load_raw(conn).values() for f in fs]
    dupes.hash_missing(conn, todo, workers=2, log=lambda *_: None)


def analyze_now(tmp_path, reg, conn, *, hash_it=True, triage_db=None, pinned=None):
    if hash_it:
        _hash_all(tmp_path, reg, conn)
    az.check_finals(conn, registry_rows(reg), tmp_path / "mirror", workers=2, log=lambda *_: None)
    pin = tmp_path / "pinned.txt"
    pin.write_text("\n".join(pinned or []))
    return az.analyze(conn, reg, triage_db=triage_db or tmp_path / "none.sqlite3", auto_db=tmp_path / "none2.sqlite3",
                      seg_db=tmp_path / "none3.sqlite3", psf_dir=tmp_path / "nopsf", pinned_file=pin,
                      mirror_root=tmp_path / "mirror", now=NOW)


def cli_args(tmp_path, *sub):
    """Argv for `cli.main` with every path inside tmp_path."""
    return ["--registry", str(tmp_path / "registry.sqlite3"), "--work-dir", str(tmp_path / "work"),
            "--seg-db", str(tmp_path / "noseg.sqlite3"), "--psf-dir", str(tmp_path / "nopsf"),
            "--mirror-root", str(tmp_path / "mirror"), "--no-throttle", *sub]


def work_db(tmp_path, conn) -> None:
    """Hand the CLI the inventory `_mk` built (the CLI opens <work-dir>/storage_triage.sqlite3)."""
    (tmp_path / "work").mkdir(exist_ok=True)
    conn.commit()
    conn.execute("VACUUM INTO ?", (str(tmp_path / "work" / "storage_triage.sqlite3"),))


@pytest.fixture
def copies(tmp_path):
    a, b = tmp_path / LAB / "cell", tmp_path / SHARE / "cell"
    for d in (a, b):
        _raw(d)
    return a, b


def a1_csv(tmp_path, reg, conn, **kw) -> Path:
    A = analyze_now(tmp_path, reg, conn, **kw)
    out = tmp_path / "report0"
    write_report(A, out, {r["dataset_key"]: r for r in registry_rows(reg)}, {})
    assert [s for s in A.suggestions if s.tier == "A1"], "fixture should produce an A1 row"
    # the PI acts on tier A1 alone here: the keeper has no row of its own, so only
    # `kept_copy` names it (the old verify never looked at that column)
    lines = (out / "suggestions.csv").read_text().splitlines()
    a1 = out / "suggestions_a1.csv"
    a1.write_text("\n".join([lines[0], *[ln for ln in lines[1:] if ln.startswith("A1,")]]) + "\n")
    return a1


# ---------------------------------------------------------------- ST-01 / ST-09: verify re-checks the keeper
def _verify(tmp_path, csv_path):
    with pytest.raises(SystemExit) as e:
        cli.main(cli_args(tmp_path, "verify", str(csv_path), "--workers", "2"))
    return e.value.code


def test_verify_passes_when_nothing_changed(tmp_path, copies, capsys):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    csv_path = a1_csv(tmp_path, reg, conn)
    work_db(tmp_path, conn)
    assert _verify(tmp_path, csv_path) == 0
    assert "rows still hold" in capsys.readouterr().out


def test_verify_fails_when_the_kept_copy_was_deleted(tmp_path, copies, capsys):
    """The ST-01 incident: the keeper's raw file is gone, the redundant copy is the last one."""
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    csv_path = a1_csv(tmp_path, reg, conn)
    work_db(tmp_path, conn)
    (a / "cell_MMStack_Pos0.ome.tif").unlink()
    assert _verify(tmp_path, csv_path) == 1
    out = capsys.readouterr().out
    assert "KEEPER CHANGED" in out and "must NOT be deleted" in out


def test_verify_fails_when_a_kept_file_was_rewritten(tmp_path, copies, capsys):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    csv_path = a1_csv(tmp_path, reg, conn)
    work_db(tmp_path, conn)
    f = a / "cell_MMStack_Pos0.ome.tif"
    f.write_bytes(np.random.default_rng(99).bytes(f.stat().st_size))     # same size, new content
    os.utime(f, ns=(1, 1))
    assert _verify(tmp_path, csv_path) == 1
    assert "KEEPER CHANGED" in capsys.readouterr().out


def test_verify_fails_when_the_kept_copy_left_the_registry(tmp_path, copies, capsys):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    csv_path = a1_csv(tmp_path, reg, conn)
    work_db(tmp_path, conn)
    reg.execute("DELETE FROM datasets WHERE dataset_key=?", (str(a),))
    reg.commit()
    assert _verify(tmp_path, csv_path) == 1
    assert "no longer in the registry" in capsys.readouterr().out


def test_verify_runs_one_analysis(tmp_path, copies, monkeypatch):
    """ST-09: verify used to analyse twice (the first result was thrown away)."""
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    csv_path = a1_csv(tmp_path, reg, conn)
    work_db(tmp_path, conn)
    calls = []
    real = cli._run_analysis
    monkeypatch.setattr(cli, "_run_analysis", lambda *x: calls.append(1) or real(*x))
    _verify(tmp_path, csv_path)
    assert len(calls) == 1


def test_report_refuses_a_deleted_keeper_stat_guard(tmp_path, copies):
    """Analysis alone (the `report` path) stats the keeper's raw files too."""
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    _hash_all(tmp_path, reg, conn)
    (a / "cell_MMStack_Pos0.ome.tif").unlink()          # inventory + hashes still claim it exists
    A = analyze_now(tmp_path, reg, conn, hash_it=False)
    assert not [s for s in A.suggestions if s.tier == "A1"]
    assert any("missing or unreadable" in n for n in A.notes)


def test_report_would_offer_the_copy_without_the_stat_guard(tmp_path, copies, monkeypatch):
    """Negative control for the test above."""
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    _hash_all(tmp_path, reg, conn)
    (a / "cell_MMStack_Pos0.ome.tif").unlink()
    monkeypatch.setattr(dupes, "keeper_file_problems", lambda *_a, **_k: [])
    A = analyze_now(tmp_path, reg, conn, hash_it=False)
    assert [s for s in A.suggestions if s.tier == "A1"]


# ---------------------------------------------------------------- ST-03: aliased copies
def test_symlinked_copy_is_not_a_duplicate(tmp_path):
    a = tmp_path / LAB / "cell"
    _raw(a)
    b = tmp_path / SHARE / "cell"
    b.parent.mkdir(parents=True)
    b.symlink_to(a, target_is_directory=True)
    reg, conn = _mk(tmp_path, [a, b])
    A = analyze_now(tmp_path, reg, conn)
    assert not [s for s in A.suggestions if s.rule == "DUP1"]
    assert any("overlaps the kept copy" in n for n in A.notes)


def test_hardlink_farm_is_not_a_duplicate(tmp_path):
    a, b = tmp_path / LAB / "cell", tmp_path / SHARE / "cell"
    _raw(a)
    b.mkdir(parents=True)
    os.link(a / "cell_MMStack_Pos0.ome.tif", b / "cell_MMStack_Pos0.ome.tif")
    (b / "AcqSettings.txt").write_text("x")
    reg, conn = _mk(tmp_path, [a, b])
    A = analyze_now(tmp_path, reg, conn)
    assert not [s for s in A.suggestions if s.rule == "DUP1"]
    assert any("same file (device/inode)" in n for n in A.notes)


def test_aliases_would_be_duplicates_without_the_guard(tmp_path, monkeypatch):
    """Negative control: identical hashes alone make an alias look like a redundant copy."""
    a, b = tmp_path / LAB / "cell", tmp_path / SHARE / "cell"
    _raw(a)
    b.parent.mkdir(parents=True)
    b.symlink_to(a, target_is_directory=True)
    reg, conn = _mk(tmp_path, [a, b])
    monkeypatch.setattr(dupes, "alias_problem", lambda *_a, **_k: None)
    A = analyze_now(tmp_path, reg, conn)
    assert {s.dataset_key for s in A.suggestions if s.rule == "DUP1"} == {str(b)}


def test_real_copies_on_separate_storage_still_pass_the_guard(tmp_path, copies):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    A = analyze_now(tmp_path, reg, conn)
    assert {s.dataset_key for s in A.suggestions if s.rule == "DUP1"} == {str(b)}


# ---------------------------------------------------------------- ST-02: every zarr channel store
def _zarr_dataset(raw_dir: Path, prefix: str, channels=("GFP_488", "mScarlet_561"), seeds=(1, 2)):
    for i, (ch, seed) in enumerate(zip(channels, seeds, strict=True)):
        store = raw_dir / f"{prefix}_{ch}.ome.zarr"
        (store / "0").mkdir(parents=True, exist_ok=True)
        (store / "0/.zarray").write_text("{}")
        (store / "0/0.0.0").write_bytes(np.random.default_rng(seed).bytes(int(1.5 * 2**20) + 4096 * i))
    return raw_dir / f"{prefix}_{channels[0]}.ome.zarr/0/0.0.0"


def _zarr_registry(tmp_path, entries):
    """entries: [(synthetic leaf_dir, master_file)] -> registry with no deskew output."""
    reg = sqlite3.connect(tmp_path / "registry.sqlite3")
    reg.row_factory = sqlite3.Row
    reg.executescript("""
    CREATE TABLE datasets (dataset_key TEXT PRIMARY KEY, root TEXT, leaf_dir TEXT, master_file TEXT,
      has_legacy_decon INTEGER DEFAULT 0, discovered_at TEXT, signal_flag TEXT, expected_timepoints INTEGER,
      actual_timepoints INTEGER, decon_psf TEXT, master_file_fingerprint TEXT, decon_params TEXT);
    CREATE TABLE stage_status (dataset_key TEXT, stage TEXT, status TEXT, started_at TEXT, finished_at TEXT,
      output_path TEXT, ticket_path TEXT, error_message TEXT, consecutive_failures INTEGER DEFAULT 0,
      PRIMARY KEY (dataset_key, stage));""")
    for leaf, master in entries:
        reg.execute("INSERT INTO datasets VALUES (?,?,?,?,0,?,?,?,?,NULL,NULL,NULL)",
                    (str(leaf), str(leaf.parent), str(leaf), str(master), OLD, "ok", 1, 1))
        for st in ("deskew", "mip_encode"):
            reg.execute("INSERT INTO stage_status VALUES (?,?,?,?,?,NULL,NULL,NULL,0)",
                        (str(leaf), st, "done", OLD, "2026-09-01T00:00:00+00:00"))
    reg.commit()
    conn = connect(tmp_path / "triage_work.sqlite3")
    scan_all(conn, registry_rows(reg), tmp_path / "mirror", workers=2, log=lambda *_: None)
    return reg, conn


def test_scan_inventories_every_channel_store(tmp_path):
    raw_dir = tmp_path / LAB
    master = _zarr_dataset(raw_dir, "cell_001")
    _zarr_dataset(raw_dir, "cell_002", seeds=(5, 6))          # another sample beside it: not ours
    row = {"leaf_dir": str(raw_dir / "cell_001"), "master_file": str(master), "final_path": None}
    scan = scan_dataset(row, tmp_path / "mirror", set())
    stores = {rel.split("/", 1)[0] for tree, rel, *_ in scan.files if tree == "master"}
    assert stores == {"cell_001_GFP_488.ome.zarr", "cell_001_mScarlet_561.ome.zarr"}


def _two_zarr_copies(tmp_path, seeds_b=(1, 2)):
    ma = _zarr_dataset(tmp_path / LAB, "cell_001")
    mb = _zarr_dataset(tmp_path / SHARE, "cell_001", seeds=seeds_b)
    return _zarr_registry(tmp_path, [(tmp_path / LAB / "cell_001", ma), (tmp_path / SHARE / "cell_001", mb)])


def test_zarr_copy_with_all_channels_identical_lists_every_store(tmp_path):
    reg, conn = _two_zarr_copies(tmp_path)
    A = analyze_now(tmp_path, reg, conn)
    got = sorted(Path(s.path).name for s in A.suggestions if s.rule == "DUP1")
    assert got == ["cell_001_GFP_488.ome.zarr", "cell_001_mScarlet_561.ome.zarr"]
    assert all(str(tmp_path / SHARE) in s.path for s in A.suggestions)


def test_zarr_copy_differing_only_in_channel_two_is_not_redundant(tmp_path):
    """ST-02: channel 0 matches, channel 1 does not. Hashing channel 0 alone called this a copy."""
    reg, conn = _two_zarr_copies(tmp_path, seeds_b=(1, 99))
    A = analyze_now(tmp_path, reg, conn)
    assert not [s for s in A.suggestions if s.rule == "DUP1"]


def test_channel_zero_only_inventory_would_call_it_redundant(tmp_path, monkeypatch):
    """Negative control: with only the master store inventoried the differing copy passes."""
    monkeypatch.setattr("backfill.storage_triage.inventory.channel_stores", lambda m: [m])
    reg, conn = _two_zarr_copies(tmp_path, seeds_b=(1, 99))
    A = analyze_now(tmp_path, reg, conn)
    assert [s for s in A.suggestions if s.rule == "DUP1"]


# ---------------------------------------------------------------- ST-04: viewer zarr only with TIFFs for every channel
def _viewer_store(leaf: Path, n_c: int):
    store = leaf / "viewer/cell_dsr.ome.zarr"
    (store / "0").mkdir(parents=True)
    (store / "0/0.0").write_bytes(b"\x01" * 2 * 2**20)
    (store / ".opym_live.json").write_text(json.dumps({"state": "complete", "n_t": 2, "n_c": n_c, "done": []}))


def _a3(tmp_path, n_tiff_channels, n_c_zarr=2, zarr_final=False):
    d = tmp_path / LAB / "cell"
    _raw(d)
    reg, conn = _mk(tmp_path, [d])
    final = d / "processed_tiff_series_split/Decon/DSR_decon"
    shutil.rmtree(final)
    _series(final, channels=n_tiff_channels, times=2)
    _viewer_store(d, n_c_zarr)
    if zarr_final:
        shutil.rmtree(final)
        z = d / "final.ome.zarr"
        (z / "0").mkdir(parents=True)
        (z / "0/.zarray").write_text(json.dumps({"shape": [4, 4], "chunks": [2, 2]}))
        for i in range(2):
            for j in range(2):
                (z / f"0/{i}.{j}").write_bytes(np.random.default_rng(i * 2 + j).bytes(40))
        reg.execute("UPDATE stage_status SET output_path=? WHERE stage='deskew'", (str(z),))
        reg.commit()
    scan_all(conn, registry_rows(reg), tmp_path / "mirror", workers=2, refresh=True, log=lambda *_: None)
    return analyze_now(tmp_path, reg, conn)


def test_viewer_zarr_offered_when_tiffs_cover_every_channel(tmp_path):
    A = _a3(tmp_path, n_tiff_channels=2)
    assert [s for s in A.suggestions if s.tier == "A3"]


def test_viewer_zarr_held_when_a_channel_has_no_tiffs(tmp_path):
    A = _a3(tmp_path, n_tiff_channels=1)
    assert not [s for s in A.suggestions if s.tier == "A3"]
    assert any("no TIFF frames for channel(s) [1]" in n for n in A.notes)


def test_viewer_zarr_held_when_final_is_zarr_only(tmp_path):
    """output_format=ome-zarr: the viewer zarr may be the only DSR copy."""
    A = _a3(tmp_path, n_tiff_channels=2, zarr_final=True)
    assert not [s for s in A.suggestions if s.tier == "A3"]
    assert any("may be the only DSR copy" in n for n in A.notes)


def _meta_store(tmp_path, attrs, shape=None, name=".zattrs"):
    store = tmp_path / "viewer/cell_dsr.ome.zarr"
    (store / "0").mkdir(parents=True)
    (store / name).write_text(json.dumps(attrs))
    if shape:
        (store / "0/.zarray").write_text(json.dumps({"shape": shape}))
    return store


_OMERO2 = {"omero": {"channels": [{"label": "a"}, {"label": "b"}]},
           "multiscales": [{"axes": [{"name": "c"}, {"name": "z"}]}]}


def test_viewer_channel_count_falls_back_to_omero_metadata(tmp_path):
    """ST-04: a backfill export without .opym_live.json is judged on its own channel list."""
    store = _meta_store(tmp_path, _OMERO2, shape=[2, 8])
    assert az.viewer_hold_reason(store, {0, 1}) is None
    assert "channel(s) [1]" in az.viewer_hold_reason(store, {0})


def test_viewer_channel_count_reads_zarr_v3_ome_attributes(tmp_path):
    store = _meta_store(tmp_path, {"attributes": {"ome": _OMERO2}}, shape=[2, 8], name="zarr.json")
    assert az.viewer_hold_reason(store, {0, 1}) is None


def test_viewer_channel_count_unknown_or_inconsistent_is_still_held(tmp_path):
    bare = tmp_path / "a/cell_dsr.ome.zarr"
    (bare / "0").mkdir(parents=True)
    assert "unknown" in az.viewer_hold_reason(bare, {0, 1})                   # no metadata at all
    clash = _meta_store(tmp_path / "b", _OMERO2, shape=[3, 8])                # omero says 2, array has 3
    assert "unknown" in az.viewer_hold_reason(clash, {0, 1, 2})
    empty = _meta_store(tmp_path / "c", {"omero": {"channels": []}})
    assert "unknown" in az.viewer_hold_reason(empty, {0, 1})


def test_viewer_zarr_offered_for_zarr_only_final_without_the_guard(tmp_path, monkeypatch):
    """Negative control: without the TIFF check the zarr-only dataset's viewer copy is offered."""
    monkeypatch.setattr(az, "viewer_hold_reason", lambda *_a, **_k: None)
    A = _a3(tmp_path, n_tiff_channels=2, zarr_final=True)
    assert [s for s in A.suggestions if s.tier == "A3"]


# ---------------------------------------------------------------- ST-05: scan age
def test_scan_ages_reports_hours_and_unscanned(tmp_path, copies):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    conn.execute("UPDATE scans SET scanned_at=? WHERE dataset_key=?", (old, str(a)))
    conn.commit()
    ages, unscanned = az.scan_ages(conn, [str(a), str(b), "/never/scanned"])
    assert 239 < ages[str(a)] < 241 and ages[str(b)] < 1 and unscanned == ["/never/scanned"]


def _age_scans(tmp_path, conn, days):
    conn.execute("UPDATE scans SET scanned_at=?", ((datetime.now(UTC) - timedelta(days=days)).isoformat(),))
    work_db(tmp_path, conn)


def test_report_refuses_a_stale_inventory(tmp_path, copies):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    _age_scans(tmp_path, conn, 10)
    with pytest.raises(SystemExit) as e:
        cli.main(cli_args(tmp_path, "report", "--verdict-db", str(tmp_path / "v.sqlite3")))
    assert "stale inventory" in str(e.value)
    assert not (tmp_path / "v.sqlite3").exists()


def test_report_with_allow_stale_labels_the_verdict_rows(tmp_path, copies):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    _age_scans(tmp_path, conn, 10)
    cli.main(cli_args(tmp_path, "report", "--allow-stale", "--verdict-db", str(tmp_path / "v.sqlite3")))
    reasons = [r[0] for r in sqlite3.connect(tmp_path / "v.sqlite3").execute("SELECT reasons FROM verdict")]
    assert reasons and all("STALE" in r for r in reasons)
    (rep,) = [d for d in (tmp_path / "work").iterdir() if d.is_dir()]
    assert json.loads((rep / "summary.json").read_text())["stale_datasets"] == 2


def test_report_on_a_fresh_inventory_needs_no_flag(tmp_path, copies):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    work_db(tmp_path, conn)
    cli.main(cli_args(tmp_path, "report", "--verdict-db", str(tmp_path / "v.sqlite3")))
    reasons = [r[0] for r in sqlite3.connect(tmp_path / "v.sqlite3").execute("SELECT reasons FROM verdict")]
    assert reasons and not any("STALE" in r for r in reasons)


def test_verdict_db_schema_is_unchanged(tmp_path, copies):
    """opym-dashboard reads this table: columns must not move."""
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    A = analyze_now(tmp_path, reg, conn)
    write_verdicts(A, tmp_path / "v.sqlite3", stale=[str(a)])
    cols = [r[1] for r in sqlite3.connect(tmp_path / "v.sqlite3").execute("PRAGMA table_info(verdict)")]
    assert cols == ["dataset_key", "rule", "action", "reasons", "suggested_bytes", "generated_at"]


# ---------------------------------------------------------------- ST-06: interior frame
def test_blank_interior_frame_is_not_verified(tmp_path):
    """The Globus blank-output class: first and last frames fine, a middle one all zeros."""
    rows = _series(tmp_path, channels=2, times=5)
    _tif(tmp_path / "cell_C1_T002.tif", v=0)
    ok, detail = verify_tiff_series(rows, tmp_path, 5)
    assert not ok and "blank" in detail and "channel 1" in detail


def test_check_version_was_bumped_so_old_results_are_redone():
    assert CHECK_VERSION >= 4


# ---------------------------------------------------------------- ST-07: protected datasets keep intermediates
def test_protected_dataset_gets_no_intermediate_suggestions(tmp_path):
    d = tmp_path / LAB / "cell"
    _raw(d)
    reg, conn = _mk(tmp_path, [d])
    assert [s for s in analyze_now(tmp_path, reg, conn).suggestions if s.tier == "A2"]     # K0: offered
    A = analyze_now(tmp_path, reg, conn, pinned=[str(d)])
    assert A.verdicts[str(d)].rule == "P1"
    assert not [s for s in A.suggestions if s.tier in ("A2", "A3")]
    assert any("protected (P1)" in n for n in A.notes)


# ---------------------------------------------------------------- ST-10: report dirs are not overwritten
def test_two_reports_never_share_a_directory(tmp_path):
    first = cli._unique_dir(tmp_path, "2026-11-07_120000")
    first.mkdir()
    second = cli._unique_dir(tmp_path, "2026-11-07_120000")
    assert second != first and second.name == "2026-11-07_120000-2"


def test_report_directory_name_carries_the_time(tmp_path, copies):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    work_db(tmp_path, conn)
    for _ in range(2):
        cli.main(cli_args(tmp_path, "report", "--verdict-db", str(tmp_path / "v.sqlite3")))
    dirs = sorted(d.name for d in (tmp_path / "work").iterdir() if d.is_dir())
    assert len(dirs) == 2 and all(len(n.split("_")[1][:6]) == 6 for n in dirs)


# ---------------------------------------------------------------- ST-11: label conflicts with partial copies
def _labels_db(path: Path, labels: dict[str, str]):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE triage (dataset_key TEXT PRIMARY KEY, label TEXT NOT NULL, updated_at TEXT NOT NULL, source TEXT)")
    for k, lab in labels.items():
        c.execute("INSERT INTO triage VALUES (?,?,?,?)", (k, lab, "2026-10-01T00:00:00+00:00", "human"))
    c.commit()
    c.close()


def _partial_copy_pair(tmp_path):
    """a (delete-labelled) shares ONE 2 MiB file with b (keep-labelled) but that is only a third of
    a's raw bytes, below the 50 % needed to be a duplicate candidate."""
    a, b = tmp_path / LAB / "cell", tmp_path / SHARE / "cell_partial"
    _raw(a, extra=[("cell_MMStack_Pos0_1.ome.tif", 2 * 2**20, 7), ("cell_MMStack_Pos0_2.ome.tif", 2 * 2**20, 8)])
    _raw(b, extra=[("cell_MMStack_Pos0_9.ome.tif", 5 * 2**20, 9)])
    reg, conn = _mk(tmp_path, [a, b])
    assert not dupes.candidate_pairs(dupes.load_raw(conn))
    _labels_db(tmp_path / "triage.sqlite3", {str(a): "delete", str(b): "keep"})
    return a, b, reg, conn


def test_smaller_partial_copy_labelled_keep_blocks_the_delete(tmp_path):
    a, b, reg, conn = _partial_copy_pair(tmp_path)
    _hash_every_raw_file(tmp_path, reg, conn)          # the shared file is hashed in both datasets
    A = analyze_now(tmp_path, reg, conn, hash_it=False, triage_db=tmp_path / "triage.sqlite3")
    assert A.verdicts[str(a)].rule == "R1" and any("labelled keep" in r for r in A.verdicts[str(a)].reasons)
    assert not [s for s in A.suggestions if s.tier == "B"]


def test_delete_goes_through_when_the_partial_copy_is_not_a_sibling_check(tmp_path, monkeypatch):
    """Negative control: with only the 50 % clusters consulted the keep label is invisible."""
    monkeypatch.setattr(dupes, "sharing_pairs", lambda *_a, **_k: set())
    a, b, reg, conn = _partial_copy_pair(tmp_path)
    _hash_every_raw_file(tmp_path, reg, conn)
    A = analyze_now(tmp_path, reg, conn, hash_it=False, triage_db=tmp_path / "triage.sqlite3")
    assert A.verdicts[str(a)].rule == "D1"


def test_unhashed_partial_copy_does_not_link(tmp_path):
    """Only a hashed shared file links two datasets; the same pair unhashed stays a D1."""
    a, b, reg, conn = _partial_copy_pair(tmp_path)
    A = analyze_now(tmp_path, reg, conn, hash_it=False, triage_db=tmp_path / "triage.sqlite3")
    assert A.verdicts[str(a)].rule == "D1"


def test_same_size_files_link_only_on_equal_hashes():
    rf = lambda k: dupes.RawFile(k, "leaf", "f", 2 * 2**20, 1)               # noqa: E731
    raw = {"a": [rf("a")], "b": [rf("b")], "c": [rf("c")]}
    digest = {"a": "x", "b": "y", "c": "x"}
    assert dupes.sharing_pairs(raw, lambda f: digest[f.dataset_key]) == {("a", "c")}
    assert not dupes.sharing_pairs(raw, lambda f: None)                      # unhashed: no evidence, no link
    assert not dupes.sharing_pairs(raw, lambda f: "x" if f.dataset_key == "a" else None)   # one side unhashed


def test_unrelated_unhashed_zarr_datasets_of_equal_chunk_size_do_not_block_a_delete(tmp_path):
    """Review T06 A.1: equal chunk sizes alone linked a human delete to an unrelated keep."""
    one = ("GFP_488",)
    ma = _zarr_dataset(tmp_path / LAB, "cell_001", channels=one, seeds=(1,))
    mb = _zarr_dataset(tmp_path / SHARE, "cell_002", channels=one, seeds=(2,))      # same size, other bytes
    a, b = tmp_path / LAB / "cell_001", tmp_path / SHARE / "cell_002"
    # one more chunk each, of different sizes, so the pair stays under the 50 % candidate bar (and unhashed)
    (ma.parent / "0.0.1").write_bytes(np.random.default_rng(3).bytes(4 * 2**20))
    (mb.parent / "0.0.1").write_bytes(np.random.default_rng(4).bytes(5 * 2**20))
    reg, conn = _zarr_registry(tmp_path, [(a, ma), (b, mb)])
    _labels_db(tmp_path / "triage.sqlite3", {str(a): "delete", str(b): "keep"})
    assert not dupes.candidate_pairs(dupes.load_raw(conn))
    A = analyze_now(tmp_path, reg, conn, triage_db=tmp_path / "triage.sqlite3")
    assert A.verdicts[str(a)].rule == "D1", A.verdicts[str(a)].reasons
    assert A.verdicts[str(b)].rule == "P1"
    assert [s for s in A.suggestions if s.tier == "B" and s.dataset_key == str(a)]


# ---------------------------------------------------------------- ST-12: throttle
def test_throttle_applies_nice_and_idle_io(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "nice", lambda n: calls.append(("nice", n)))
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/ionice")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(tuple(cmd)) or subprocess.CompletedProcess(cmd, 0))
    args = cli.argparse.Namespace(no_throttle=False)
    assert cli.throttle(args) == ["nice +10", "ionice idle"]
    assert ("nice", 10) in calls and any(c[:3] == ("/usr/bin/ionice", "-c", "3") for c in calls if len(c) > 2)


def test_throttle_opt_out(monkeypatch):
    monkeypatch.setattr(os, "nice", lambda n: pytest.fail("must not renice"))
    assert cli.throttle(cli.argparse.Namespace(no_throttle=True)) == []


def test_rate_limiter_spaces_reads():
    now = [0.0]
    slept = []
    lim = dupes.RateLimiter(100, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    for _ in range(3):
        lim.acquire(100)                                     # 100 B at 100 B/s = 1 s each
    assert slept == [1.0, 1.0] or slept == pytest.approx([1.0, 1.0])


def test_hash_command_accepts_a_rate_cap(tmp_path, copies):
    a, b = copies
    reg, conn = _mk(tmp_path, [a, b])
    work_db(tmp_path, conn)
    cli.main(cli_args(tmp_path, "hash", "--workers", "2", "--max-mbps", "500"))
    n = sqlite3.connect(tmp_path / "work" / "storage_triage.sqlite3").execute("SELECT COUNT(*) FROM hashes").fetchone()[0]
    assert n >= 2
