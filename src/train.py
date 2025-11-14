# src/train_full_final.py
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# -------------------------
# CONFIG
# -------------------------
DATA_TRAIN_DIR = "data/split/train"
DATA_VAL_DIR = "data/split/val"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-4

BEST_MODEL_NAME = os.path.join(MODEL_DIR, "bes_kannada_ocr_final.h5")
FINAL_MODEL_NAME = os.path.join(MODEL_DIR, "final_kannada_ocr_final.h5")
LOG_DIR = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.json")

# -------------------------
# DATA GENERATORS
# -------------------------
print("📂 Loading data generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=12,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.08,
    zoom_range=0.08,
    brightness_range=(0.8, 1.2),
    horizontal_flip=False,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    DATA_TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    DATA_VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"🔢 Classes found: {num_classes}")
print(f"📦 Train samples: {train_generator.samples}, Val samples: {val_generator.samples}")

# -------------------------
# CLASS WEIGHTS
# -------------------------
print("⚖️ Computing class weights...")
if hasattr(train_generator, "classes"):
    classes_arr = train_generator.classes
    class_labels = np.unique(classes_arr)
    cw = compute_class_weight(class_weight="balanced", classes=class_labels, y=classes_arr)
    class_weight = {int(i): float(w) for i, w in enumerate(cw)}
    print("✅ Class weights prepared.")
else:
    class_weight = None
    print("⚠️ train_generator.classes not found — skipping class weights.")

# -------------------------
# MODEL
# -------------------------
print("🧠 Building model (MobileNetV2 backbone)...")
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet")
base_model.trainable = False

inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inp, outputs=out)
loss_fn = CategoricalCrossentropy(label_smoothing=0.05)
model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss=loss_fn, metrics=["accuracy"])
model.summary()

# -------------------------
# CALLBACKS
# -------------------------
callbacks = [
    ModelCheckpoint(BEST_MODEL_NAME, monitor="val_accuracy", verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor="val_loss", patience=12, verbose=1, restore_best_weights=True),
    TensorBoard(log_dir=LOG_DIR)
]

# -------------------------
# TRAINING
# -------------------------
print("🚀 Step1: Train top dense layers (backbone frozen)")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

print("🔁 Unfreezing top of backbone for fine-tuning...")
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=INITIAL_LR * 0.1), loss=loss_fn, metrics=["accuracy"])

print("🚀 Step2: Fine-tuning with lower lr")
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=history1.epoch[-1] + 1 if hasattr(history1, "epoch") else 30,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# -------------------------
# SAVE FINAL MODEL
# -------------------------
print("💾 Saving final model and training history...")
model.save(FINAL_MODEL_NAME)

# Merge histories
def merge_histories(h1, h2):
    merged = {}
    for k in set(h1.history.keys()).union(h2.history.keys()):
        merged[k] = h1.history.get(k, []) + h2.history.get(k, [])
    return merged

history = merge_histories(history1, history2)
with open(HISTORY_FILE, "w") as f:
    json.dump(history, f)

# -------------------------
# PLOT LOSS & ACCURACY
# -------------------------
plt.figure(figsize=(10, 5))
epochs = range(1, len(history["accuracy"]) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, history["accuracy"], label="Train Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history["loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

os.makedirs("outputs", exist_ok=True)
plt.tight_layout()
plot_path = os.path.join("outputs", "training_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"📊 Training curve saved to: {plot_path}")

# -------------------------
# FINAL METRICS
# -------------------------
print("📈 Evaluating final metrics...")
val_generator.reset()
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")
acc = accuracy_score(y_true, y_pred)

print(f"\n✅ FINAL METRICS:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\n✅ Best model: {BEST_MODEL_NAME}")
print(f"✅ Final model: {FINAL_MODEL_NAME}")
print(f"📈 TensorBoard logs: {LOG_DIR}")
print(f"📝 Training history saved to: {HISTORY_FILE}")





'''# src/train_improved.py
import os
import json
import math
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight

# -------------------------
# CONFIG
# -------------------------
DATA_TRAIN_DIR = "data/split/train"
DATA_VAL_DIR = "data/split/val"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)         # MobileNetV2 default-ish size; larger helps transfer learning
BATCH_SIZE = 32               # increase if you have GPU/RAM
EPOCHS = 120                  # will stop early with EarlyStopping
INITIAL_LR = 1e-4

BEST_MODEL_NAME = os.path.join(MODEL_DIR, "best_kannada_ocr_v2.h5")
FINAL_MODEL_NAME = os.path.join(MODEL_DIR, "final_kannada_ocr_v2.h5")
LOG_DIR = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

# -------------------------
# DATA AUGMENTATION
# -------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # MobileNetV2 preprocessing (RGB)
    rotation_range=12,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.08,
    zoom_range=0.08,
    brightness_range=(0.8, 1.2),
    horizontal_flip=False,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# -------------------------
# DATA GENERATORS
# -------------------------
print("📂 Loading data generators...")
train_generator = train_datagen.flow_from_directory(
    DATA_TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATA_VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"🔢 Classes found: {num_classes}")
print(f"📦 Train samples: {train_generator.samples}, Val samples: {val_generator.samples}")

# -------------------------
# COMPUTE CLASS WEIGHTS (helps with class imbalance)
# -------------------------
print("⚖️ Computing class weights...")
if hasattr(train_generator, "classes"):
    classes_arr = train_generator.classes
    class_labels = np.unique(classes_arr)
    cw = compute_class_weight(class_weight="balanced", classes=class_labels, y=classes_arr)
    class_weight = {int(i): float(w) for i, w in enumerate(cw)}
    print("✅ Class weights prepared.")
else:
    class_weight = None
    print("⚠️ train_generator.classes not found — skipping class weights.")

# -------------------------
# BUILD MODEL: Transfer Learning
# -------------------------
print("🧠 Building model (MobileNetV2 backbone)...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze backbone initially

inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inp, outputs=out)

loss_fn = CategoricalCrossentropy(label_smoothing=0.05)  # small label smoothing helps generalization

model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# CALLBACKS
# -------------------------
callbacks = [
    ModelCheckpoint(BEST_MODEL_NAME, monitor="val_accuracy", verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor="val_loss", patience=12, verbose=1, restore_best_weights=True),
    TensorBoard(log_dir=LOG_DIR)
]

# -------------------------
# STEP 1: Train top layers
# -------------------------
print("🚀 Step1: Train top dense layers (backbone frozen)")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# -------------------------
# STEP 2: Fine-tune some backbone layers
# -------------------------
print("🔁 Unfreezing top of backbone for fine-tuning...")

# Unfreeze last blocks of the base model for fine-tuning
base_model.trainable = True
# freeze earlier layers, unfreeze last N
fine_tune_at = int(len(base_model.layers) * 0.7)  # unfreeze last 30%
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# lower lr for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR * 0.1),
    loss=loss_fn,
    metrics=["accuracy"]
)

print("🚀 Step2: Fine-tuning with lower lr")
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,          # will stop early if no improvement
    initial_epoch=history1.epoch[-1] + 1 if hasattr(history1, "epoch") else 30,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# -------------------------
# SAVE FINAL MODEL
# -------------------------
print("💾 Saving final model...")
model.save(FINAL_MODEL_NAME)
print("✅ Done. Best model:", BEST_MODEL_NAME)
print("✅ Final model:", FINAL_MODEL_NAME)
print("📈 Use TensorBoard logs:", LOG_DIR)'''

'''import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from datetime import datetime

# ========================
# PATHS & BASIC CONFIG
# ========================
data_dir = r"D:\kannada_ocr_project\data\raw\KannadaHnd\Kannada\Hnd\Img"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

num_classes = 657
img_height, img_width = 64, 64
batch_size = 32
epochs = 50

# ========================
# DATA AUGMENTATION
# ========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ========================
# MODEL ARCHITECTURE
# ========================
def build_model(input_shape=(img_height, img_width, 1), num_classes=657):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.4),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.4),

        Conv2D(512, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.5),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model()
model.summary()

# ========================
# COMPILE MODEL
# ========================
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ========================
# CALLBACKS
# ========================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath=os.path.join(model_dir, 'best_kannada_ocr.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1),
    TensorBoard(log_dir=os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S")))
]

# ========================
# TRAINING
# ========================
steps_per_epoch = len(train_gen)
validation_steps = len(val_gen)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# ========================
# SAVE FINAL MODEL
# ========================
model.save(os.path.join(model_dir, "final_kannada_ocr.h5"))
print("\n✅ Training complete! Best model saved as 'models/best_kannada_ocr.h5'")
'''
