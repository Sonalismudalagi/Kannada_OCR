from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model("models/best_kannada_ocr_final.h5")

# Load test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = "data/split/val"   # path where test images are stored, each class in its own folder

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# Optional: Get predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Count correct & incorrect
correct = np.sum(predicted_classes == true_classes)
incorrect = np.sum(predicted_classes != true_classes)

print(f"✔️ Correctly Predicted: {correct}")
print(f"❌ Incorrectly Predicted: {incorrect}")
print(f"📊 Total Images Tested: {len(true_classes)}")
