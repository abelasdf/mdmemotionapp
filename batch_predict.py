test_dir = "archive/test/happy"

import onnxruntime as ort
import numpy as np
import cv2
import os

# Modell laden
session = ort.InferenceSession("model/model.onnx")
input_name = session.get_inputs()[0].name

EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

# 👇 Dein lokaler Pfad zu den Bildern
test_dir = "/Users/abel/Downloads/archive/test/happy"
true_label = "happiness"

# Bilder iterieren
for filename in os.listdir(test_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(test_dir, filename)

        # Bild vorbereiten
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Fehler beim Laden von {filename}")
            continue
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32)
        img -= 128.0
        img = img.reshape(1, 1, 64, 64)

        # Vorhersage
        output = session.run(None, {input_name: img})[0][0]
        predicted = EMOTION_LABELS[np.argmax(output)]

        # Ergebnis anzeigen
        korrekt = "✔️" if predicted == true_label else "❌"
        print(f"{filename}: {predicted} (expected: {true_label}) {korrekt}")