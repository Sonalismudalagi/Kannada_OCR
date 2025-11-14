import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# --------------------
# Paths
# --------------------
RAW_DATA_DIR = "data/raw/GoodImgBmp"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

IMG_SIZE = 64

# --------------------
# Load dataset
# --------------------
X, y = [], []
label_to_idx, idx_to_label = {}, {}
current_label = 0

print("🔍 Scanning dataset folders...")

for folder in sorted(os.listdir(RAW_DATA_DIR)):
    folder_path = os.path.join(RAW_DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    if folder not in label_to_idx:
        label_to_idx[folder] = current_label
        idx_to_label[current_label] = folder
        current_label += 1

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if img_file.lower().endswith(".db"):  # skip Thumbs.db
            continue
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label_to_idx[folder])
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

print(f"✅ Loaded dataset: X={X.shape}, y={y.shape}, num_classes={len(label_to_idx)}")

# --------------------
# Drop singleton classes
# --------------------
class_counts = Counter(y)
valid_indices = [i for i, label in enumerate(y) if class_counts[label] > 1]

X = X[valid_indices]
y = y[valid_indices]

print(f"✅ After removing singleton classes: X={X.shape}, y={y.shape}, num_classes={len(np.unique(y))}")

# --------------------
# Train/test split (safe now)
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape}, {y_train.shape}")
print(f"✅ Test set: {X_test.shape}, {y_test.shape}")

# --------------------
# Save processed data
# --------------------
np.savez(os.path.join(PROCESSED_DIR, "kannada_dataset.npz"),
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test)

np.save(os.path.join(PROCESSED_DIR, "label_to_idx.npy"), label_to_idx, allow_pickle=True)
np.save(os.path.join(PROCESSED_DIR, "idx_to_label.npy"), idx_to_label, allow_pickle=True)

print("💾 Saved processed dataset at data/processed/kannada_dataset.npz")
print("💾 Saved label_to_idx.npy and idx_to_label.npy")
