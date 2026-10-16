"""decon_v2_trial: where the driver puts each arm's output, and what the variants file says."""
import json
from pathlib import Path

import numpy as np
import pytest

from psf_tools import decon_v2_trial as T


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    (tmp_path / "psfs.json").write_text(json.dumps({k: str(tmp_path / f"{k}.tif") for k in
                                                     ("synL_488", "synL_561", "syn_488", "syn_561", "synL_sa0_488", "synL_sa0_561")}))
    (tmp_path / "Cell_004").mkdir()
    return tmp_path


def test_dsr_tif_follows_the_drivers_layout(ws):
    arms = T.arm_table(T.load_psfs(), "Cell_004")
    st = ws / "Cell_004" / "stage"
    assert T.dsr_tif("Cell_004", arms["nodecon"], 0) == st / "DSR" / "Cell_004_C0_T023.tif"
    assert T.dsr_tif("Cell_004", arms["prod"], 1) == st / "Decon_prod" / "DSR" / "Cell_004_C1_T023.tif"
    # padded arms decon in a sibling dir of the stage, never inside it
    assert T.dsr_tif("Cell_004", arms["pad"], 0) == ws / "Cell_004" / "pad24_pad" / "Decon_pad" / "DSR" / "Cell_004_C0_T023.tif"
    # an arm with its own stage dir (denoised input) nests under that
    assert T.dsr_tif("Cell_004", arms["n2v_prod"], 0) == ws / "Cell_004" / "stage_n2v" / "Decon_n2v_prod" / "DSR" / "Cell_004_C0_T023.tif"
    assert T.dsr_tif("Cell_004", arms["synL_bg"], 0).parts[-4:-2] == ("pad24_synL_bg", "Decon_synL_bg")
    # the combined arm: denoised + haze-floor-subtracted input, large synthetic PSF, padding
    comb = arms["n2v_synL_bg"]
    assert comb["stage_dir"].endswith("stage_n2v_bg") and comb["pad_z"] == 24 and len(comb["psf_paths"]) == 2
    assert T.dsr_tif("Cell_004", comb, 0).parts[-4:-2] == ("pad24_n2v_synL_bg", "Decon_n2v_synL_bg")
    # the two "view" arms are plain DSRs of their own stage dirs
    assert T.dsr_tif("Cell_004", arms["n2v_only"], 0) == ws / "Cell_004" / "stage_n2v" / "DSR" / "Cell_004_C0_T023.tif"
    assert T.dsr_tif("Cell_004", arms["floor_only"], 1) == ws / "Cell_004" / "stage_floor" / "DSR" / "Cell_004_C1_T023.tif"


def test_corrected_floor_arms_and_the_shifted_poisson_arms(ws):
    arms = T.arm_table(T.load_psfs(), "Cell_004")
    # production leaves background None: PetaKit itself turns that into 100 (XR_decon_data_wrapper.m:219), so an explicit
    # background=100 arm would be bit-identical to it and is deliberately absent
    assert arms["prod"]["background"] is None and not any(n.endswith("_bias") or n == "bias" for n in arms)
    assert arms["prod_bg100_check"]["background"] == 100.0   # the identity control, not a candidate
    assert arms["prod_bg93"]["background"] == arms["synL_bg93"]["background"] == T.SHIFTED_POISSON_BG == 93.0
    for n in ("n2v_synL_bgv2", "n2v_synL_bgv2o6", "n2v_synL_bgv2o10", "n2v_dark_synL_bgv2", "synL_bg93"):
        assert arms[n]["pad_z"] == 24 and len(arms[n]["psf_paths"]) == 2
        assert n.startswith("synL_bg93") or arms[n]["background"] is None
    # the floors of the corrected geometry live in their own directories, never the pre-fix ones
    assert arms["n2v_synL_bgv2"]["stage_dir"].endswith("stage_n2v_bgv2")
    assert arms["n2v_synL_bgv2o6"]["stage_dir"].endswith("stage_n2v_bgv2_o6")
    assert arms["n2v_dark_synL_bgv2"]["stage_dir"].endswith("stage_n2v_dark_bgv2")
    assert arms["n2v_rep_synL_bgv2"]["stage_dir"].endswith("stage_n2v_rep_bgv2") and arms["n2v_rep_synL_bgv2"]["background"] is None
    assert arms["floor_onlyv2"]["stage_dir"].endswith("stage_floorv2")
    assert arms["floor_onlyv2o6"]["stage_dir"].endswith("stage_floorv2_o6") and arms["floor_onlyv2o10"]["decon"] is False
    for n in ("synL_bg", "n2v_synL_bg", "floor_only"):
        assert "mislabelled" in arms[n]["superseded"]
    assert not any("superseded" in arms[n] for n in arms if n not in ("synL_bg", "n2v_synL_bg", "floor_only"))


def test_floor_dir_names_do_not_collide_with_the_pre_fix_directories():
    names = {T.floor_dir_name(s, o, v) for s in ("stage", "stage_n2v", "stage_n2v_dark") for o in (3.0, 6.0) for v in (False, True)}
    assert len(names) == 12 and not names & {"stage_bg", "stage_floor", "stage_n2v_bg"}


def test_floor_stage_writes_in_the_staged_layout_and_keeps_the_shape(ws):
    import tifffile
    src = ws / "Cell_004" / "stage"
    src.mkdir()
    rng = np.random.default_rng(0)
    vol = (150 + rng.normal(0, 3, (12, 64, 48))).astype(np.uint16)       # (nz, ny, nx)
    tifffile.imwrite(src / "Cell_004_C0_T023.tif", vol)
    out = T.floor_stage("Cell_004")
    assert out.name == "stage_bgv2"
    got = tifffile.imread(out / "Cell_004_C0_T023.tif")
    assert got.shape == vol.shape and got.dtype == np.uint16 and (got <= vol).all() and got.min() >= 100 - 30
    view = tifffile.imread(T.floor_stage("Cell_004", view=True) / "Cell_004_C0_T023.tif")
    assert view.shape == vol.shape and view.mean() < vol.mean()
    with pytest.raises(SystemExit):
        T.floor_stage("Cell_004", src="stage_nothing")


def test_production_arm_is_the_production_config():
    from opym.decon_config import DECON_DAMP_FACTOR, DECON_EDGE_EROSION, DECON_WIENER_ALPHA
    prod = T.arm_table({})["prod"]
    assert (prod["alpha"], prod["damp"], prod["iters"], prod["rl_method"]) == (DECON_WIENER_ALPHA, DECON_DAMP_FACTOR, 2, "omw")
    assert "pad_z" not in prod and "edge_erosion" not in prod  # inherits the top-level DECON_EDGE_EROSION, no padding


def test_variants_file_carries_production_geometry_and_psf_per_channel(ws):
    vj = json.loads(T.make_variants("Cell_004", ["nodecon", "prod", "synL"]).read_text())
    assert vj["interp_method"] == "linear" and vj["edge_erosion"] == 3 and vj["z_step_um"] == 0.5 and vj["dz_psf"] == 0.1
    assert [v["name"] for v in vj["variants"]] == ["nodecon", "prod", "synL"]
    assert len(vj["variants"][2]["psf_paths"]) == 2 and vj["variants"][2]["pad_z"] == 24


def test_unknown_or_unavailable_arms_are_refused(ws):
    with pytest.raises(SystemExit):
        T.make_variants("Cell_004", ["nope"])
    (ws / "psfs.json").write_text("{}")
    with pytest.raises(SystemExit):
        T.make_variants("Cell_004", ["synL"])  # no synthetic PSF yet


def test_compare_and_zyx_orientation():
    a = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    assert T.compare(a, a)["exact"]
    c = T.compare(a, a + 1)
    assert not c["exact"] and c["max_abs_diff"] == 1.0
    yxz = np.transpose(a, (1, 2, 0))
    np.testing.assert_array_equal(T.as_zyx(yxz, a.shape), a)
    with pytest.raises(ValueError):
        T.as_zyx(np.zeros((2, 3, 5)), a.shape)


def test_dark_stage_uses_each_channels_own_pattern_and_corrects_the_training_frames(ws, monkeypatch):
    import tifffile
    monkeypatch.setattr(T, "TRIAL_CELLS", {"Cell_004": 23})
    rng = np.random.default_rng(0)
    ny, nx, nz = 30, 24, 80
    pats = [rng.normal(0, 1.0, (ny, nx)), rng.normal(0, 1.0, (ny, nx))]  # C0 and C1 are different cameras' patterns
    st, tr = ws / "Cell_004" / "stage", ws / "Cell_004" / "n2v_train"
    st.mkdir(), tr.mkdir()
    for ch in (0, 1):
        for d, t in ((st, 23), (tr, 3)):
            tifffile.imwrite(d / f"Cell_004_C{ch}_T{t:03d}.tif", (rng.normal(100, 4.5, (nz, ny, nx)) + pats[ch][None]).astype(np.float32))
    out = T.stage_dark_cell("Cell_004")
    assert out.name == "stage_dark" and (ws / "Cell_004" / "n2v_train_dark" / "Cell_004_C1_T003.tif").is_file()
    for ch in (0, 1):
        for d in (out, ws / "Cell_004" / "n2v_train_dark"):
            f = next(d.glob(f"Cell_004_C{ch}_T*.tif"))
            got = tifffile.imread(f)
            assert got.dtype == np.float32 and abs(got.mean() - 100) < 0.5            # the offset is kept
            own = np.corrcoef(got.mean(0).ravel(), pats[ch].ravel())[0, 1]
            other = np.corrcoef(got.mean(0).ravel(), pats[1 - ch].ravel())[0, 1]
            assert abs(own) < 0.3 and abs(other) < 0.3                                  # pattern removed, nothing else's added
    assert T._channel_of("Cell_001_002_C1_T020.tif") == 1
    with pytest.raises(ValueError):
        T._channel_of("nochannel.tif")


def test_identity_check_compares_prod_with_an_explicit_background_100(ws):
    import tifffile
    arms = T.arm_table(T.load_psfs(), "Cell_004")
    for a in ("prod", "prod_bg100_check"):
        T.make_variants("Cell_004", [a])
    T.make_variants("Cell_004", ["prod", "prod_bg100_check"])
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 500, (6, 50, 50), dtype=np.uint16)
    for name, v in (("prod", vol), ("prod_bg100_check", vol)):
        for c in (0, 1):
            f = T.dsr_tif("Cell_004", arms[name], c)
            f.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(f, v)
    assert all(r["identical"] and r["max_abs_diff"] == 0 for r in T.identity_check("Cell_004").values())
    tifffile.imwrite(T.dsr_tif("Cell_004", arms["prod_bg100_check"], 1), vol + 1)   # negative control: one count off is caught
    assert not T.identity_check("Cell_004")["C1"]["identical"]
