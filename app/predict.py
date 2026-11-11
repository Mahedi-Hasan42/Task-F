import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "final_mobilenetv2.keras"
CLASSES_PATH = ROOT / "data" / "classes.json"
THRESHOLD_PATH = ROOT / "artifacts" / "threshold.json"
IMG_SIZE = 224


class Classifier:
    def __init__(self, threshold=None):
        self.model = keras.models.load_model(MODEL_PATH, compile=False)
        with open(CLASSES_PATH, encoding="utf-8") as f:
            self.classes = json.load(f)["classes"]
        if threshold is None:
            with open(THRESHOLD_PATH, encoding="utf-8") as f:
                threshold = json.load(f)["threshold"]
        self.threshold = threshold

    def _predict(self, img):
        t0 = time.perf_counter()
        x = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])   
        x = preprocess_input(x)                          
        probs = self.model(x[None], training=False).numpy()[0]
        ms = (time.perf_counter() - t0) * 1000
        top = np.argsort(probs)[::-1][:3]
        return {
            "label": self.classes[int(top[0])],
            "confidence": float(probs[top[0]]),
            "top3": [(self.classes[int(i)], float(probs[i])) for i in top],
            "ms": ms,
        }

    def predict_file(self, path):
        data = tf.io.read_file(str(path))
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        return self._predict(img)

    def predict_frame(self, bgr_frame):
        rgb = bgr_frame[:, :, ::-1].copy()
        return self._predict(tf.convert_to_tensor(rgb))