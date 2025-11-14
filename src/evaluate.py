'''import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================================
# CONFIGURATION
# =========================================
VAL_DIR = r"D:\kannada_ocr_project\data\split\val"  # Path to validation folder
MODEL_PATH = r"D:\kannada_ocr_project\models\best_kannada_ocr_v2.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================================
# LOAD MODEL
# =========================================
print("📦 Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.\n")

# =========================================
# PREPARE VALIDATION DATA
# =========================================
datagen = ImageDataGenerator(rescale=1.0/255.0)

val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================================
# MAKE PREDICTIONS
# =========================================
print("🔍 Predicting on validation set...")
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

# =========================================
# METRICS
# =========================================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\n📊 MODEL EVALUATION METRICS 📊")
print(f"✅ Accuracy : {accuracy * 100:.2f}%")
print(f"🎯 Precision: {precision * 100:.2f}%")
print(f"📈 Recall   : {recall * 100:.2f}%")
print(f"🏆 F1-Score : {f1 * 100:.2f}%")

# Optional: detailed per-class report
report = classification_report(y_true, y_pred, digits=3)
os.makedirs("outputs", exist_ok=True)
report_path = os.path.join("outputs", "classification_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n📝 Detailed per-class report saved to: {report_path}")
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Load best model
# =========================
def load_best_model(model_dir="models"):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("❌ No .h5 model found in models/ folder.")
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    print(f"📦 Loading model from {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully.\n")
    return model

# =========================
# Main evaluation
# =========================
if __name__ == "__main__":
    model = load_best_model()

    # detect input shape (RGB or grayscale)
    input_shape = model.input_shape
    color_mode = "rgb" if input_shape[-1] == 3 else "grayscale"

    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32

    val_dir = os.path.join("data", "split", "val")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"❌ Validation directory not found: {val_dir}")

    print(f"📂 Loading validation data from: {val_dir}")
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        color_mode=color_mode,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("🔍 Predicting on validation data...")
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    # =========================
    # Compute metrics
    # =========================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n==============================")
    print(f"✅ Overall Accuracy : {acc*100:.2f}%")
    print(f"🎯 Precision         : {prec*100:.2f}%")
    print(f"🔁 Recall            : {rec*100:.2f}%")
    print(f"🏆 F1-Score          : {f1*100:.2f}%")
    print("==============================\n")
