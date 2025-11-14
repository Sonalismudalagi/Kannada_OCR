import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === CONFIG ===
TRAIN_DIR = "data/split/train"
MODEL_DIR = "models"

# === Load latest trained model ===
def load_best_model(model_dir=MODEL_DIR):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("❌ No model files found in 'models/' folder.")
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = os.path.join(model_dir, model_files[0])
    print(f"📦 Loading model from: {latest_model}")
    model = load_model(latest_model)
    print("✅ Model loaded successfully.\n")
    return model

# === Data generator (no augmentation) ===
datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False  # Important: don’t shuffle for evaluation
)

# === Load model ===
model = load_best_model()

# === Evaluate ===
print("🔍 Evaluating training accuracy...")
pred_probs = model.predict(train_gen, verbose=1)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = train_gen.classes

# === Compute accuracy ===
acc = accuracy_score(true_classes, pred_classes)
conf_mat = confusion_matrix(true_classes, pred_classes)
report = classification_report(true_classes, pred_classes, zero_division=0)

# === Count correct & incorrect ===
total = len(true_classes)
correct = int(acc * total)
incorrect = total - correct

print("\n==============================")
print(f"📊 Total training images: {total}")
print(f"✅ Correct predictions: {correct}")
print(f"❌ Incorrect predictions: {incorrect}")
print(f"🎯 Training accuracy: {acc * 100:.2f}%")
print("==============================\n")

# === Optional: Save report ===
os.makedirs("outputs", exist_ok=True)
with open("outputs/train_evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Training Accuracy: {acc * 100:.2f}%\n")
    f.write(f"Correct Predictions: {correct}\n")
    f.write(f"Incorrect Predictions: {incorrect}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("📝 Detailed report saved to: outputs/train_evaluation_report.txt")
