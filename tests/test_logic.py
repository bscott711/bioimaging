import re
from pathlib import Path

FLUOROPHORES = {
    r'mng|ng|gfp|calcein': 'GFP/mNG (488)',
    r'msca|msc|scarlet|568': 'mScarlet/CF568 (561)',
    r'cf647|647|sir': 'CF647/SiR (640)',
    r'hoechst|dapi|violet': 'Hoechst/Violet (405)',
    r'beads|yg': 'YG Beads (488 nm excitation)',
    r'cav': 'Calcein Violet (cav) [Excited by 405nm]'
}
# Removed FLM from fluorophores because it's a marker, not a distinct color channel?
# Wait, in the artifact, FLM IS in the dictionary:
# r'flm': 'Fetal Liver Macrophage (flm)'
# If FLM is in the dictionary, then len(found) for this folder is 3!
# But the user said: "Should only be mNG and mSc. So this would be Ch0 top and Ch1 top".
# This means the user EXCLUDES FLM from the channel count!

def parse_name(name):
    name = name.lower()
    found = []
    for pattern, std in FLUOROPHORES.items():
        if re.search(pattern, name):
            found.append(std)
    return found

print(parse_name("20260402_py_FLM_2XFyve_mSca_mem_NG"))
