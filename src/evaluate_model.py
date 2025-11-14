import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import json

# ========================
# CONFIG
# ========================
MODEL_PATH = "models/best_kannada_ocr_final.h5"
DATA_DIR = r"D:\kannada_ocr_project\data\raw\KannadaHnd\Kannada\Hnd\Img"  # adjust if needed
IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32

# ========================
# LOAD MODEL
# ========================
print("📦 Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# ========================
# LOAD VALIDATION DATA
# ========================
print("📂 Preparing validation dataset...")
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ========================
# EVALUATE MODEL
# ========================
print("\n📊 Evaluating model performance...")
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n✅ Validation Accuracy: {val_acc*100:.2f}%")
print(f"✅ Validation Loss: {val_loss:.4f}")

# ========================
# DETAILED METRICS
# ========================
print("\n🔍 Generating detailed classification report...")
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

report = classification_report(val_generator.classes, y_pred, output_dict=True)
print("\n✅ Precision, Recall, F1-score summary:")
print(json.dumps(report['weighted avg'], indent=4))

# Save to a file (for report inclusion)
os.makedirs("outputs", exist_ok=True)
with open("outputs/classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

print("\n📄 Detailed metrics saved to outputs/classification_report.json")

# ========================
# TRAINING HISTORY GRAPH
# ========================
# During training, if you saved history using pickle, load and plot it:
HISTORY_PATH = "models/training_history.pkl"
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "rb") as f:
        history = pickle.load(f)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/accuracy_graph.png")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/loss_graph.png")
    plt.show()

    print("📊 Accuracy and loss graphs saved in outputs/ folder.")
else:
    print("⚠️ No training history file found (models/training_history.pkl).")
