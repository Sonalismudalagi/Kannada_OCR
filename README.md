🚀 Kannada Handwritten Character Recognition (CNN + MobileNetV2 + Flask)

A deep learning–based OCR system to recognize handwritten Kannada characters.


📝 Project Overview

This work implements a deep learning–based OCR model for recognizing 611 Kannada handwritten characters.
The model uses:

MobileNetV2 as backbone

Custom dense layers for classification

A Flask web app for real-time character prediction

Preprocessing, augmentation, and training pipeline built from scratch

The trained model achieves ~96% accuracy on validation data.

🎯 Key Features

✔ 611-class Kannada handwritten character recognition

✔ MobileNetV2 transfer learning for high accuracy & faster convergence

✔ Real-time prediction via a Flask web interface

✔ Automatic preprocessing (resize, normalize, RGB correction)

✔ Training logs, model checkpoints, and TensorBoard support

✔ Accuracy, Precision, Recall, and F1-score evaluation

📁 Project Structure
kannada_ocr_project/
│
├── data/
│   ├── raw/
│   └── split/
│       ├── train/
│       └── val/
│
├── models/
│   ├── best_kannada_ocr.h5
│   └── final_kannada_ocr.h5
│
├── src/
│   ├── folder_to_unicode.py
│   └── predict.py
│   └── train.py
├── web/
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── result.html
│
├── outputs/
│   ├── training_curve.png
│   └── predictions/
│
└── README.md

🧠 Model Architecture (Simplified)

Input (224×224×3)

MobileNetV2 Backbone (pretrained on ImageNet)

GlobalAveragePooling2D

Dense (512, ReLU)

BatchNorm → Dropout

Dense (num_classes, Softmax)

📦 Setup & Installation
1️⃣ Clone Repository
git clone https://github.com/your-username/kannada-ocr-project.git
cd kannada-ocr-project

2️⃣ Install Dependencies
pip install -r requirements.txt


3️⃣ Run the Training Script
python train.py

4️⃣ Run the Web App
cd web
python app.py


The app runs at:

👉 http://127.0.0.1:5000/

🖥️ Usage
1. Upload Image

Upload any Kannada handwritten character image (PNG/JPG).

2. Model Predicts

Predicted class folder

Unicode character

Confidence score

Visualization saved in outputs/predictions/

3. Output Shown on Browser

Shows image + prediction.

📊 Results
Metric	Score
Accuracy	~0.95
Precision	High
Recall	High
F1 Score	High

📌 Metrics automatically calculated using scikit-learn.

🖼️ Screenshots
🔹 Web Interface

(Upload your image)

![Web UI](<img width="961" height="1021" alt="Screenshot 2025-11-12 103719" src="https://github.com/user-attachments/assets/9b697d97-7ea5-4ef4-a5c3-a3bc13a51c70" />)

🔹 Prediction Result
![Prediction](<img width="961" height="1021" alt="Screenshot 2025-11-12 103928" src="https://github.com/user-attachments/assets/78772882-4d67-4272-ae2e-e7442c9b976d" />)


🧩 Technologies Used
Languages

Python

Frameworks & Libraries

TensorFlow

Keras

MobileNetV2

Flask

OpenCV

NumPy

scikit-learn

Matplotlib

Tools

VS Code

TensorBoard

Git & GitHub

🧪 Future Work

Add support for multi-character word recognition

Implement sequence modeling (BiLSTM + CTC)

Deploy model on cloud (AWS / GCP)

Convert to mobile app using TensorFlow Lite
