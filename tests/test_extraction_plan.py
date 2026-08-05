from pathlib import Path

from psf_tools.extraction_plan import get_extraction_plan

if __name__ == "__main__":
    # Manual smoke-check against a real dataset path -- not a pytest
    # assertion, kept for ad hoc inspection. Guarded so pytest collection
    # doesn't execute a real-filesystem-dependent side effect.
    p = Path(
        "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/"
        "20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0_47.ome.tif"
    )
    print(get_extraction_plan(p))
