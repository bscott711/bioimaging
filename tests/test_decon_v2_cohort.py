"""decon_v2_cohort / decon_v2_trial per-cell specs: the cohort table, the interpolation chosen by the control gate, which frames N2V trains on."""
import json
from pathlib import Path

import pytest

from psf_tools import decon_v2_cohort as C
from psf_tools import decon_v2_trial as T

TABLE = {
    "9999_Cell_A": dict(cell="Cell_A", t=12, membrane_ch=0, z_step_um=0.5, n_timepoints=100, timepoints=list(range(100)),
                        reference=dict(kind="tiff", path="/x/ref.tif"), raw={"GFP_488": "/x/A_GFP_488.ome.zarr", "mScarlet_561": "/x/A_mScarlet_561.ome.zarr"}),
    "9999_Cell_B": dict(cell="Cell_B", t=1, membrane_ch=0, z_step_um=None, n_timepoints=2, timepoints=[0, 1],
                        reference=dict(kind="zarr", path="/x/B.zarr", t=1, c=0), raw={"GFP_488": "/x/B_GFP_488.ome.zarr", "mScarlet_561": "/x/B_mScarlet_561.ome.zarr"}),
}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    f = tmp_path / "cells.json"
    f.write_text(json.dumps(TABLE))
    monkeypatch.setattr(T, "WORKSPACE", tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "psfs.json").write_text(json.dumps({k: str(tmp_path / f"{k}.tif") for k in ("synL_488", "synL_561", "syn_488", "syn_561", "synL_sa0_488", "synL_sa0_561")}))
    monkeypatch.setattr("opym.metadata.parse_zarr_z_step_from_store", lambda store: 0.3)
    saved = dict(T.SPECS), dict(T.TRIAL_CELLS), dict(T.COHORT_ID)
    ids = T.load_cohort(f)
    yield ids, tmp_path
    T.SPECS.clear(); T.SPECS.update(saved[0]); T.TRIAL_CELLS.clear(); T.TRIAL_CELLS.update(saved[1]); T.COHORT_ID.clear(); T.COHORT_ID.update(saved[2])


def test_load_cohort_builds_membrane_only_specs_and_reads_the_z_step_from_the_store_when_the_table_has_none(cohort):
    ids, _ = cohort
    assert ids == ["9999_Cell_A", "9999_Cell_B"]
    a, b = T.spec("9999_Cell_A"), T.spec("9999_Cell_B")
    assert a.channels == (0,) and len(a.stores) == 1 and a.stores[0].name == "A_GFP_488.ome.zarr"
    assert a.z_step_um == 0.5 and b.z_step_um == 0.3                      # the table's own, else the store's (mocked 0.3)
    assert T.TRIAL_CELLS["9999_Cell_B"] == 1 and T.COHORT_ID["9999_Cell_B"] == "9999_Cell_B"


def test_variants_file_for_a_cohort_cell_has_one_channel_one_psf_and_its_own_z_step(cohort):
    T.cell_dir("9999_Cell_A").mkdir(parents=True, exist_ok=True)
    vj = json.loads(T.make_variants("9999_Cell_A", ["prod", "n2v_synL_bg"]).read_text())
    assert vj["channel_patterns"] == ["_C0_T"] and vj["z_step_um"] == 0.5
    comb = [v for v in vj["variants"] if v["name"] == "n2v_synL_bg"][0]
    assert len(comb["psf_paths"]) == 1                                    # the driver checks one PSF per channel pattern


def test_the_interpolation_found_by_the_control_gate_applies_to_every_arm_but_the_controls(cohort):
    sp = T.spec("9999_Cell_A")
    T.SPECS["9999_Cell_A"] = T.CellSpec(sp.t, sp.cohort_id, sp.stores, sp.z_step_um, sp.reference, sp.channels, sp.extra_ts, "cubic")
    arms = T.arm_table(T.load_psfs(), "9999_Cell_A")
    assert arms["n2v_synL_bg"]["interp_method"] == "cubic" and arms["synL"]["interp_method"] == "cubic"
    assert "interp_method" not in arms["prod"] and arms["prod_cubic"]["interp_method"] == "cubic"   # the controls stay what they are
    assert "interp_method" not in T.arm_table(T.load_psfs(), "9999_Cell_B")["n2v_synL_bg"]           # no override: the pipeline default


def test_dsr_tif_follows_t_and_defaults_to_the_segmentation_timepoint(cohort):
    arms = T.arm_table(T.load_psfs(), "9999_Cell_A")
    assert T.dsr_tif("9999_Cell_A", arms["prod"], 0).name == "9999_Cell_A_C0_T012.tif"
    assert T.dsr_tif("9999_Cell_A", arms["prod"], 0, t=14).name == "9999_Cell_A_C0_T014.tif"


def test_unknown_cell_is_refused_with_the_way_to_add_it():
    with pytest.raises(SystemExit):
        T.spec("nope")


def test_training_frames_are_spread_through_the_series_and_always_include_the_segmentation_timepoint():
    assert C.train_timepoints(23, list(range(138))) == [0, 23, 68, 103, 137]
    assert C.train_timepoints(63, list(range(100))) == [0, 25, 50, 63, 99]
    assert C.train_timepoints(1, [0, 1]) == [0, 1]                        # a short series: all of it
    for t in (0, 50, 99):
        got = C.train_timepoints(t, list(range(100)))
        assert t in got and len(got) == 5 and got == sorted(got)
