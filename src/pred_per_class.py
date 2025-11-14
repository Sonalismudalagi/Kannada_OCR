import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# ========================
# CONFIGURATION
# ========================
MODEL_PATH = "models/best_kannada_ocr_final.h5"   # change if needed
DATA_DIR = "data/raw/KannadaHnd/Kannada/Hnd/Img"  # your dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# LOAD MODEL
# ========================
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.\n")

# ========================
# VALIDATION GENERATOR
# ========================
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ========================
# PREDICTIONS
# ========================
print("🔍 Predicting on validation data...")
Y_pred = model.predict(val_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes

# ========================
# OVERALL ACCURACY
# ========================
overall_acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Validation Accuracy: {overall_acc * 100:.2f}%")

# ========================
# PER-CLASS ACCURACY
# ========================
class_labels = list(val_generator.class_indices.keys())
correct_per_class = np.zeros(len(class_labels))
total_per_class = np.zeros(len(class_labels))

for i in range(len(y_true)):
    total_per_class[y_true[i]] += 1
    if y_pred[i] == y_true[i]:
        correct_per_class[y_true[i]] += 1

class_accuracy = {}
for i, label in enumerate(class_labels):
    if total_per_class[i] > 0:
        acc = correct_per_class[i] / total_per_class[i]
        class_accuracy[label] = acc * 100

# ========================
# SAVE CLASS-WISE ACCURACY
# ========================
class_acc_df = pd.DataFrame(list(class_accuracy.items()), columns=["Class", "Accuracy (%)"])
class_acc_csv_path = os.path.join(OUTPUT_DIR, "class_accuracy.csv")
class_acc_df.to_csv(class_acc_csv_path, index=False, encoding='utf-8')
print(f"📄 Class-wise accuracy saved to: {class_acc_csv_path}")

# ========================
# CLASSIFICATION REPORT
# ========================
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
report_df.to_csv(report_csv_path, encoding='utf-8')
print(f"📊 Detailed classification report saved to: {report_csv_path}")

# ========================
# ACCURACY GRAPH
# ========================
plt.figure(figsize=(15, 6))
plt.bar(range(len(class_accuracy)), list(class_accuracy.values()), color='skyblue')
plt.title("Per-Class Validation Accuracy", fontsize=16)
plt.xlabel("Class Index", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xticks(range(0, len(class_accuracy), max(1, len(class_accuracy)//20)),
           list(class_accuracy.keys())[::max(1, len(class_accuracy)//20)],
           rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

graph_path = os.path.join(OUTPUT_DIR, "accuracy_vs_class.png")
plt.tight_layout()
plt.savefig(graph_path)
plt.close()
print(f"📈 Accuracy vs Class graph saved to: {graph_path}")

# ========================
# SUMMARY
# ========================
print("\n✅ Evaluation Complete!")
print(f"🧾 Overall Accuracy: {overall_acc * 100:.2f}%")
print("📊 Files saved in 'outputs/' folder:")
print(f" - {class_acc_csv_path}")
print(f" - {report_csv_path}")
print(f" - {graph_path}")
