import json
import os

def load_folder_to_unicode():
    """Loads folder_to_unicode mapping from JSON file."""
    mapping_path = os.path.join("src", "folder_to_unicode.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("❌ folder_to_unicode.json not found in src/. Run build_mapping.py first.")
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        folder_to_unicode = json.load(f)
    
    return folder_to_unicode
