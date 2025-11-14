# src/build_idx_to_unicode.py
import os
import numpy as np
from pathlib import Path

# import your folder -> unicode mapping
#from src.folder_to_unicode import folder_to_unicode
from src.utils import load_folder_to_unicode
folder_to_unicode = load_folder_to_unicode()

IDX_TO_LABEL_PATH = "data/processed/idx_to_label.npy"   # created by preprocessing.py
OUT_PATH = "data/processed/idx_to_unicode.npy"

if not os.path.exists(IDX_TO_LABEL_PATH):
    raise SystemExit(f"❌ Cannot find {IDX_TO_LABEL_PATH}. Run preprocessing.py first.")

# load idx -> folder (should be a dict like {0: "Sample001", 1: "Sample002", ...})
idx_to_label = np.load(IDX_TO_LABEL_PATH, allow_pickle=True).item()

idx_to_unicode = {}
missing_folders = []

for idx, folder in idx_to_label.items():
    # idx might be string or int; ensure int key
    idx_int = int(idx)
    # look up folder in user mapping
    if folder in folder_to_unicode:
        idx_to_unicode[idx_int] = folder_to_unicode[folder]
    else:
        # fallback: leave folder name as label and log that mapping is missing
        idx_to_unicode[idx_int] = folder
        missing_folders.append(folder)

# save mapping
os.makedirs("data/processed", exist_ok=True)
np.save(OUT_PATH, idx_to_unicode, allow_pickle=True)
print(f"✅ Saved idx->unicode mapping to {OUT_PATH}")

if missing_folders:
    print("⚠️ Missing folder→unicode mappings for these folders (please add to src/folder_to_unicode.py):")
    for f in sorted(set(missing_folders))[:200]:
        print("  ", f)
    print(f"({len(missing_folders)} missing).")
else:
    print("✅ All folders mapped to Unicode characters.")
