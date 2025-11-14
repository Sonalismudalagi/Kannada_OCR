import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 📂 Paths
DATA_DIR = "data\\raw\\KannadaHnd\\Kannada\\Hnd\\Img"
SPLIT_DIR = "data/split"

TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")

# 🧹 Cleanup old split
if os.path.exists(SPLIT_DIR):
    shutil.rmtree(SPLIT_DIR)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ⚙️ Parameters
VAL_RATIO = 0.2  # 20% for validation

# 📸 Split logic
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        print(f"⚠️ No images in {folder}, skipping...")
        continue

    # 🪄 Handle single-image folders by duplicating the image
    if len(images) == 1:
        print(f"🔁 Duplicating single image in {folder} to allow splitting...")
        images = images * 2  # make a copy of the same image for train/val split

    # ✂️ Split into train and val
    train_files, val_files = train_test_split(images, test_size=VAL_RATIO, random_state=42)

    # Create class directories
    os.makedirs(os.path.join(TRAIN_DIR, folder), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, folder), exist_ok=True)

    # Copy train files
    for f in train_files:
        src = os.path.join(folder_path, f if f in os.listdir(folder_path) else images[0])
        dst = os.path.join(TRAIN_DIR, folder, f)
        shutil.copy(src, dst)

    # Copy val files
    for f in val_files:
        src = os.path.join(folder_path, f if f in os.listdir(folder_path) else images[0])
        dst = os.path.join(VAL_DIR, folder, f)
        shutil.copy(src, dst)

print("\n✅ Dataset split complete!")
print(f"📁 Training data: {TRAIN_DIR}")
print(f"📁 Validation data: {VAL_DIR}")
