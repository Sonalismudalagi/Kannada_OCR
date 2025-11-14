import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("models/final_kannada_ocr_final.h5")

# Validation data
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_directory(
    "data/split/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Predictions
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print(f"\n📊 FINAL MODEL PERFORMANCE:")
print(f"✅ Accuracy :  {acc:.4f}")
print(f"✅ Precision:  {precision:.4f}")
print(f"✅ Recall   :  {recall:.4f}")
print(f"✅ F1 Score :  {f1:.4f}")
