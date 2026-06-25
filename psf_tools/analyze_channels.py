import os
import json
import re
from pathlib import Path

# Known fluorophore patterns and their standardized names
FLUOROPHORES = {
    r'mng|ng|gfp|calcein': 'GFP/mNG (488)',
    r'msca|msc|scarlet|568': 'mScarlet/CF568 (561)',
    r'cf647|647|sir': 'CF647/SiR (640)',
    r'hoechst|dapi|violet': 'Hoechst/Violet (405)',
    r'beads|yg': 'YG Beads (488 nm excitation)',
    r'cav': 'Calcein Violet (cav) [Excited by 405nm]',
    r'flm': 'Fetal Liver Macrophage (flm)'
}

def parse_name_for_fluorophores(name):
    name = name.lower()
    found = []
    for pattern, std_name in FLUOROPHORES.items():
        if re.search(pattern, name):
            found.append(std_name)
    return found

def analyze_directory(base_dir):
    base_path = Path(base_dir)
    results = []
    
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
            
        dir_name = item.name
        
        # We look for AcqSettings.txt in the subdirectories (e.g. cell_1/AcqSettings.txt)
        acq_files = list(item.glob("**/AcqSettings.txt"))
        
        config_types = []
        parsed_channels = []
        source = "None"
        
        if acq_files:
            # Parse the first AcqSettings.txt found
            try:
                with open(acq_files[0], 'r') as f:
                    data = json.load(f)
                    channels = data.get("channels", [])
                    if channels:
                        config_types = [c.get("config_", "") for c in channels]
                        if len(config_types) == 1 and config_types[0] == "All Lasers":
                            source = "Folder Name (All Lasers in AcqSettings)"
                            parsed_channels = parse_name_for_fluorophores(dir_name)
                        else:
                            source = "AcqSettings Config"
                            parsed_channels = config_types
            except Exception as e:
                source = f"Error Parsing JSON: {e}"
                parsed_channels = parse_name_for_fluorophores(dir_name)
        else:
            source = "Folder Name (No AcqSettings)"
            parsed_channels = parse_name_for_fluorophores(dir_name)
            
        results.append({
            "Folder": dir_name,
            "Source": source,
            "Raw_Configs": config_types,
            "Detected_Channels": parsed_channels
        })
        
    return results

if __name__ == "__main__":
    base_dir = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/"
    results = analyze_directory(base_dir)
    
    print(f"{'Folder Name':<50} | {'Source':<35} | {'Detected Channels'}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x['Folder']):
        chans = ", ".join(r['Detected_Channels']) if r['Detected_Channels'] else "Unknown"
        print(f"{r['Folder']:<50} | {r['Source']:<35} | {chans}")
