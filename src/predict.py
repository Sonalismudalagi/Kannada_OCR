'''import cv2
import numpy as np
from tensorflow.keras.models import load_model
#from src.folder_to_unicode import folder_to_unicode
from src.utils import load_folder_to_unicode
folder_to_unicode = load_folder_to_unicode()

# Load model
MODEL_PATH = "models/kannada_vowels.keras"
model = load_model(MODEL_PATH)

# Reverse mapping: index -> unicode char
index_to_unicode = {
    idx: unicode_char for idx, (folder, unicode_char) in enumerate(folder_to_unicode.items())
}

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0
    return img.reshape(1, 32, 32, 1)

# Example test image (change path to your test case)
test_image = "data/raw/GoodImgBmp/Sample008/img008-00001.png"

img = preprocess_image(test_image)
pred = model.predict(img)
pred_idx = np.argmax(pred)
confidence = np.max(pred)

predicted_char = index_to_unicode.get(pred_idx, "❓")

print(f"✅ Prediction: {predicted_char} (confidence={confidence:.2f})")

'''
'''import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import load_folder_to_unicode

# ---------------------------
# Load Model and Mappings
# ---------------------------
MODEL_PATH = "models/vowel_model_optimized.h5"  # update if using new dataset model
model = load_model(MODEL_PATH)

# Load folder-to-Unicode mapping
folder_to_unicode = load_folder_to_unicode()

# Reverse mapping: index -> Unicode char
index_to_unicode = {
    idx: unicode_char for idx, (_, unicode_char) in enumerate(folder_to_unicode.items())
}

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(img_path):
    """
    Reads and preprocesses a single Kannada character image.
    Converts to RGB, resizes to 64x64, and normalizes pixel values.
    """
    img = cv2.imread(img_path)  # Read in color (BGR)
    if img is None:
        raise FileNotFoundError(f"⚠️ Could not read image at {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)
    return img


# ---------------------------
# Prediction Function
# ---------------------------
def predict_character(img_path):
    """
    Predicts the Kannada character in the given image and returns
    both Unicode and confidence.
    """
    img = preprocess_image(img_path)
    pred = model.predict(img)
    pred_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    predicted_char = index_to_unicode.get(pred_idx, "❓")
    
    print(f"✅ Prediction: {predicted_char} (confidence={confidence:.2f})")
    return predicted_char, confidence

# ---------------------------
# Example Run
# ---------------------------
if __name__ == "__main__":
    # Change to your test image
    test_image = "data/raw/GoodImgBmp/Sample001/img001-00001.png"
    predict_character(test_image)'''
    
'''# src/predict.py
"""
Predict Kannada character from an image using the trained CNN model.
Usage:
    python -m src.predict path_to_image
"""

import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------------
# Configuration
# -------------------------------------
MODEL_PATH = "models/best_kannda_chars.h5"
INDEX_TO_FOLDER_JSON = "src/folder_to_unicode.json"
FOLDER_TO_UNICODE_PY = "src/folder_to_unicode.py"  # we will import dynamically
IMG_SIZE = (64, 64)

# -------------------------------------
# Import folder_to_unicode mapping
# -------------------------------------
folder_to_unicode = {}
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("folder_to_unicode", FOLDER_TO_UNICODE_PY)
    folder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(folder_module)
    folder_to_unicode = folder_module.folder_to_unicode
    print("✅ Loaded folder_to_unicode mapping from src/folder_to_unicode.py")
except Exception as e:
    print(f"⚠️ Could not load folder_to_unicode.py: {e}")

# -------------------------------------
# Load index -> folder mapping
# -------------------------------------
with open(INDEX_TO_FOLDER_JSON, "r", encoding="utf-8") as f:
    index_to_folder = json.load(f)
index_to_folder = {int(k): v for k, v in index_to_folder.items()}

# -------------------------------------
# Load model
# -------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"🚨 Model not found at {MODEL_PATH}")
    sys.exit(1)

print(f"📦 Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.\n")

# -------------------------------------
# Preprocessing functions (same as training)
# -------------------------------------
def deskew_and_denoise(image):
    img = image.copy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(255 - th)
    if coords is not None and len(coords) > 20:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.25:
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    p2, p98 = np.percentile(img, (2, 98))
    if p98 - p2 > 0:
        img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

    return img

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    img = deskew_and_denoise(img)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (64, 64, 1)
    img = np.expand_dims(img, axis=0)   # (1, 64, 64, 1)
    return img

# -------------------------------------
# Predict function
# -------------------------------------
def predict_character(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    class_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    folder_name = index_to_folder.get(class_index, "Unknown")
    unicode_char = folder_to_unicode.get(folder_name, "❓")

    print("🖼️ Image:", os.path.basename(image_path))
    print(f"📁 Predicted folder: {folder_name}")
    print(f"🔤 Kannada character: {unicode_char}")
    print(f"📊 Confidence: {confidence:.4f}")

    return folder_name, unicode_char, confidence

# -------------------------------------
# CLI execution
# -------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"🚨 File not found: {image_path}")
        sys.exit(1)

    predict_character(image_path)

'''
'''import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import importlib.util

# ========================
# LOAD UNICODE MAPPING
# ========================
def load_unicode_mapping():
    mapping_path = os.path.join("src", "folder_to_unicode.py")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("❌ folder_to_unicode.py not found in src/ folder.")
    
    spec = importlib.util.spec_from_file_location("folder_to_unicode", mapping_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ Loaded folder_to_unicode mapping from src/folder_to_unicode.py")
    return module.folder_to_unicode

# ========================
# LOAD MODEL
# ========================
def load_best_model(model_dir="models"):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("❌ No model found in models/ folder.")
    
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    
    print(f"📦 Loading model from {model_path} ...")
    model = load_model(model_path)
    print("✅ Model loaded successfully.\n")
    return model

# ========================
# PREPROCESS IMAGE
# ========================
def preprocess_image(img_path, img_height=64, img_width=64):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========================
# PREDICT CHARACTER
# ========================
def predict_character(img_path):
    folder_to_unicode = load_unicode_mapping()
    model = load_best_model()

    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    folder_name = f"Sample{predicted_index + 1:03d}"
    kannada_char = folder_to_unicode.get(folder_name, "❓ Unknown")

    print(f"🖼️ Image: {os.path.basename(img_path)}")
    print(f"📁 Predicted folder: {folder_name}")
    print(f"🔤 Kannada character: {kannada_char}")
    print(f"📊 Confidence: {confidence:.4f}")

# ========================
# MAIN ENTRY
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python -m src.predict <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

    predict_character(img_path)
'''

'''import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import importlib.util
import cv2

# ========================
# LOAD UNICODE MAPPING
# ========================
def load_unicode_mapping():
    mapping_path = os.path.join("src", "folder_to_unicode.py")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("❌ folder_to_unicode.py not found in src/ folder.")
    
    spec = importlib.util.spec_from_file_location("folder_to_unicode", mapping_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ Loaded folder_to_unicode mapping from src/folder_to_unicode.py")
    return module.folder_to_unicode

# ========================
# LOAD MODEL
# ========================
def load_best_model(model_dir="models"):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("❌ No model found in models/ folder.")
    
    # sort by modification time (latest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    
    print(f"📦 Loading model from {model_path} ...")
    model = load_model(model_path)
    print("✅ Model loaded successfully.\n")
    return model

# ========================
# PREPROCESS IMAGE
# ========================
def preprocess_image(img_path, img_height=224, img_width=224):
    # Read using OpenCV (handles any input format)
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"❌ Could not read image: {img_path}")

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize to model input size
    img = cv2.resize(img, (img_width, img_height))

    # Normalize to [0, 1]
    img = img.astype("float32") / 255.0

    # Expand dimensions to make it (1, H, W, 3)
    img = np.expand_dims(img, axis=0)

    return img

# ========================
# PREDICT CHARACTER
# ========================
def predict_character(img_path):
    folder_to_unicode = load_unicode_mapping()
    model = load_best_model()

    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    folder_name = f"Sample{predicted_index + 1:03d}"
    kannada_char = folder_to_unicode.get(folder_name, "❓ Unknown")

    print("\n===========================")
    print(f"🖼️ Image: {os.path.basename(img_path)}")
    print(f"📁 Predicted folder: {folder_name}")
    print(f"🔤 Kannada character: {kannada_char}")
    print(f"📊 Confidence: {confidence:.4f}")
    print("===========================\n")

# ========================
# MAIN ENTRY
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python -m src.predict <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

    predict_character(img_path)
'''
'''#matplotlib
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import importlib.util
import cv2
import matplotlib.pyplot as plt

# ========================
# LOAD UNICODE MAPPING
# ========================
def load_unicode_mapping():
    mapping_path = os.path.join("src", "folder_to_unicode.py")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("❌ folder_to_unicode.py not found in src/ folder.")
    
    spec = importlib.util.spec_from_file_location("folder_to_unicode", mapping_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ Loaded folder_to_unicode mapping from src/folder_to_unicode.py")
    return module.folder_to_unicode

# ========================
# LOAD MODEL
# ========================
def load_best_model(model_dir="models"):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("❌ No model found in models/ folder.")
    
    # sort by modification time (latest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    
    print(f"📦 Loading model from {model_path} ...")
    model = load_model(model_path)
    print("✅ Model loaded successfully.\n")
    return model

# ========================
# PREPROCESS IMAGE
# ========================
def preprocess_image(img_path, img_height=224, img_width=224):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"❌ Could not read image: {img_path}")

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Keep a copy of original for display
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    resized_img = cv2.resize(img, (img_width, img_height))

    # Normalize to [0, 1]
    resized_img = resized_img.astype("float32") / 255.0

    # Expand dimensions to (1, H, W, 3)
    resized_img = np.expand_dims(resized_img, axis=0)

    return resized_img, original_img

# ========================
# PREDICT CHARACTER + VISUALIZE
# ========================
def predict_character(img_path):
    folder_to_unicode = load_unicode_mapping()
    model = load_best_model()

    processed_img, original_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    folder_name = f"Sample{predicted_index + 1:03d}"
    kannada_char = folder_to_unicode.get(folder_name, "❓ Unknown")

    # Console Output
    print("\n===========================")
    print(f"🖼️ Image: {os.path.basename(img_path)}")
    print(f"📁 Predicted folder: {folder_name}")
    print(f"🔤 Kannada character: {kannada_char}")
    print(f"📊 Confidence: {confidence:.4f}")
    print("===========================\n")

    # ========================
    # VISUALIZE RESULT
    # ========================
    plt.imshow(original_img)
    plt.title(f"Predicted: {kannada_char} | Confidence: {confidence:.2f}", fontsize=14)
    plt.axis('off')

    # Create output folder if not exists
    os.makedirs("outputs/predictions", exist_ok=True)
    save_path = os.path.join("outputs/predictions", f"pred_{os.path.basename(img_path)}")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"📸 Prediction visualization saved to: {save_path}")

    plt.show()

# ========================
# MAIN ENTRY
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python -m src.predict <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

    predict_character(img_path)'''

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import importlib.util
import cv2
import matplotlib.pyplot as plt

# ========================
# LOAD UNICODE MAPPING
# ========================
def load_unicode_mapping():
    mapping_path = os.path.join("src", "folder_to_unicode.py")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("❌ folder_to_unicode.py not found in src/ folder.")
    
    spec = importlib.util.spec_from_file_location("folder_to_unicode", mapping_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ Loaded folder_to_unicode mapping from src/folder_to_unicode.py")
    return module.folder_to_unicode

# ========================
# LOAD MODEL
# ========================
def load_best_model(model_dir="models"):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("❌ No model found in models/ folder.")
    
    # Sort by modification time (latest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    
    print(f"📦 Loading model from {model_path} ...")
    model = load_model(model_path)
    print("✅ Model loaded successfully.\n")
    return model

# ========================
# PREPROCESS IMAGE
# ========================
def preprocess_image(img_path, img_height=224, img_width=224):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Could not read image: {img_path}")

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Keep a copy for display
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    resized_img = cv2.resize(img, (img_width, img_height))

    # Normalize to [0, 1]
    resized_img = resized_img.astype("float32") / 255.0

    # Expand dimensions to (1, H, W, 3)
    resized_img = np.expand_dims(resized_img, axis=0)

    return resized_img, original_img

# ========================
# PREDICT CHARACTER + VISUALIZE
# ========================
def predict_character(img_path):
    folder_to_unicode = load_unicode_mapping()
    model = load_best_model()

    processed_img, original_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    folder_name = f"Sample{predicted_index + 1:03d}"
    kannada_char = folder_to_unicode.get(folder_name, "❓ Unknown")

    # Console Output
    print("\n===========================")
    print(f"🖼️ Image: {os.path.basename(img_path)}")
    print(f"📁 Predicted folder: {folder_name}")
    print(f"🔤 Kannada character: {kannada_char}")
    print(f"📊 Confidence: {confidence:.4f}")
    print("===========================\n")

    # ========================
    # VISUALIZE RESULT
    # ========================
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_img)
    ax.axis('off')

    # Display text (in English or Kannada)
    title_text = (
        f"Image: {os.path.basename(img_path)}\n"
        f"Predicted: {kannada_char}\n"
        f"Confidence: {confidence:.2f}"
    )

    # Use a Unicode-supporting font if available (e.g. Nirmala UI)
    try:
        plt.rcParams['font.family'] = 'Nirmala UI'
    except:
        pass

    ax.set_title(title_text, fontsize=12)

    os.makedirs("outputs/predictions", exist_ok=True)
    save_path = os.path.join("outputs/predictions", f"pred_{os.path.basename(img_path)}")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"📸 Prediction visualization saved to: {save_path}")

    # Return results for web or CLI
    return {
        "image_name": os.path.basename(img_path),
        "folder": folder_name,
        "character": kannada_char,
        "confidence": float(confidence),
        "visualization_path": save_path
    }

# ========================
# MAIN ENTRY (for CLI)
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python -m src.predict <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

    result = predict_character(img_path)
    print(f"\n✅ Result:")
    print(result)
