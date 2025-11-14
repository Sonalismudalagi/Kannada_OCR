import json

# Create mapping 0 → Sample001, 1 → Sample002, … 656 → Sample657
folder_map = {str(i): f"Sample{str(i+1).zfill(3)}" for i in range(657)}

# Save as JSON
with open("folder_to_unicode.json", "w", encoding="utf-8") as f:
    json.dump(folder_map, f, ensure_ascii=False, indent=4)

print("✅ folder_to_unicode.json with 657 mappings created successfully!")
