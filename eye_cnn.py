"""CNN eye-state classifier helpers. Model expects grayscale 64x64 images."""
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_eye_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CNN model not found at {path}. Train it using train_cnn.py.")
    return load_model(path)

def predict_eye_state(model, image):
    image = cv2.resize(image, (64, 64)).astype("float32") / 255.0
    image = image.reshape(1, 64, 64, 1)
    probability_closed = float(model.predict(image, verbose=0)[0][0])
    return "closed" if probability_closed >= 0.5 else "open"
