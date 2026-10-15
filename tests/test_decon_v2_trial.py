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
