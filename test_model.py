import onnxruntime as ort
import numpy as np
import cv2

# Emotionen laut Modell
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

# Bild laden und vorbereiten
img_path = "test/face_sample.png"  # Pfad ggf. anpassen
img = cv2.imread("test/emotion_sample.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("archive/test/happy/sample1.jpg")
# → wie oben vorbereiten
# → Ergebnis prüfen

if img is None:
    raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")

img = cv2.resize(img, (64, 64))
img = img.astype(np.float32)
img -= 128.0  # Normalisierung
img = img.reshape(1, 1, 64, 64)

# Modell laden
session = ort.InferenceSession("model/model.onnx")
input_name = session.get_inputs()[0].name

# Vorhersage durchführen
output = session.run(None, {input_name: img})[0][0]

# Top Emotion finden
top_idx = np.argmax(output)
print(f"Erkannte Emotion: {EMOTION_LABELS[top_idx]}")