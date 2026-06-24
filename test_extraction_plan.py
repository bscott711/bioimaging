from pathlib import Path
from analyze_channels import analyze_directory, parse_name_for_fluorophores

def get_extraction_plan(target_path: Path):
    base_dir = target_path.parent.parent
    results = analyze_directory(str(base_dir))
    folder_name = target_path.parent.name
    
    match = next((r for r in results if r["Folder"] == folder_name), None)
    if not match: return []
    
    detected = match["Detected_Channels"]
    if not detected:
        detected = parse_name_for_fluorophores(base_dir.name)
        
    raw_configs = match["Raw_Configs"]
    
    has_488 = any("488" in c for c in detected)
    has_405 = any("405" in c for c in detected)
    has_561 = any("561" in c for c in detected)
    has_640 = any("640" in c for c in detected)
    
    plan = []
    
    if len(raw_configs) <= 1:
        if has_488: plan.append((0, True, "488nm"))
        if has_405: plan.append((0, False, "405nm"))
        if has_561: plan.append((1, True, "561nm"))
        if has_640: plan.append((1, False, "640nm"))
    else:
        for i, config in enumerate(raw_configs):
            config_lower = config.lower()
            if "488" in config_lower: plan.append((i*2 + 0, True, "488nm"))
            if "405" in config_lower: plan.append((i*2 + 0, False, "405nm"))
            if "561" in config_lower: plan.append((i*2 + 1, True, "561nm"))
            if "640" in config_lower: plan.append((i*2 + 1, False, "640nm"))
            
    return plan

p = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0_47.ome.tif")
print(get_extraction_plan(p))
