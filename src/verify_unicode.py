from src.folder_to_unicode import folder_to_unicode

total = len(folder_to_unicode)
missing = []
placeholder = []

# check total count
print(f"🔢 Total mappings found: {total}")

# find any missing or placeholder entries
for i in range(1, 658):
    folder_name = f"Sample{i:03d}"
    if folder_name not in folder_to_unicode:
        missing.append(folder_name)
    elif folder_to_unicode[folder_name] == "?":
        placeholder.append(folder_name)

if not missing:
    print("✅ All 657 folders have mappings.")
else:
    print(f"❌ Missing mappings: {missing}")

if placeholder:
    print(f"⚠️ Placeholders used for invalid/missing chars: {len(placeholder)} samples.")
else:
    print("✅ No placeholder mappings.")

print("\n✅ Verification complete!")
