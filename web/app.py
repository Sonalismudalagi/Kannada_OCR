import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues
import matplotlib.pyplot as plt

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import load_unicode_mapping, load_best_model, preprocess_image

# -------------------------
# Flask Configuration
# -------------------------
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "web", "static", "uploads")
PREDICT_FOLDER = os.path.join(os.getcwd(), "web", "static", "predictions")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # -------------------------
    # Model Prediction
    # -------------------------
    folder_to_unicode = load_unicode_mapping()
    model = load_best_model()

    processed_img, original_img = preprocess_image(file_path)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    folder_name = f"Sample{predicted_index + 1:03d}"
    kannada_char = folder_to_unicode.get(folder_name, "❓ Unknown")

    # -------------------------
    # Save prediction visualization
    # -------------------------
    plt.imshow(original_img)
    plt.axis("off")
    plt.title(f"Predicted: {kannada_char} | Confidence: {confidence:.2f}", fontsize=14)

    pred_filename = f"pred_{filename}"
    save_path = os.path.join(PREDICT_FOLDER, pred_filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Convert path for static display
    display_path = os.path.join("static", "predictions", pred_filename)

    # -------------------------
    # Render result page
    # -------------------------
    return render_template(
        "result.html",
        image_name=display_path,
        character=kannada_char,
        confidence=f"{confidence*100:.2f}%",
        folder=folder_name
    )

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
