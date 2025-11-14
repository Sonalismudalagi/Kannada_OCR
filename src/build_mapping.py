import os
import json

DATA_DIR = "data\\raw\\KannadaHnd\\Kannada\\Hnd\\Img"
MAPPING_FILE = "src/folder_to_unicode.json"  # store mapping in JSON instead of .py

# Load existing mapping if present
if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        folder_to_unicode = json.load(f)
else:
    folder_to_unicode = {}

# Scan all SampleXXX folders
for folder in os.listdir(DATA_DIR):
    if folder.startswith("Sample"):
        if folder not in folder_to_unicode:
            folder_to_unicode[folder] = "?"   # placeholder

# Save updated mapping
with open(MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(folder_to_unicode, f, ensure_ascii=False, indent=4)

print(f"✅ Mapping file updated with {len(folder_to_unicode)} entries.")
print(f"Fill in the '?' with the correct Kannada characters.")
