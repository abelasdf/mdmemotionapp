from flask import Flask, render_template, request
import onnxruntime as ort
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

# Modell laden
session = ort.InferenceSession("model/emotion_model.onnx")
input_name = session.get_inputs()[0].name

def predict_emotion(image_path):
    img = cv2.imread(image_path)  # Lädt in BGR
    if img is None:
        return "Bildfehler"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konvertiere zu RGB
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
    img = img.reshape(1, 3, 64, 64)  # Batch dimension
    output = session.run(None, {input_name: img})[0][0]
    top_idx = np.argmax(output)
    return EMOTION_LABELS[top_idx]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            prediction = predict_emotion(filepath)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)