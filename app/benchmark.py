import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 224
RUNS = 20


def bench(model_path, image_path, mobilenet_preprocess):
    model = keras.models.load_model(model_path, compile=False)

    def one_prediction():
        data = tf.io.read_file(str(image_path))
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        x = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        if mobilenet_preprocess:
            x = preprocess_input(x)
        probs = model(x[None], training=False).numpy()[0]
        return int(np.argmax(probs))

    one_prediction()                      
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        one_prediction()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "runs": RUNS,
        "median_ms": round(statistics.median(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="ekta test image er path")
    args = p.parse_args()

    result = {
        "machine": {
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
        },
        "mobilenetv2": bench(ROOT / "artifacts" / "final_mobilenetv2.keras", args.image, True),
        "scratch_cnn_v2": bench(ROOT / "artifacts" / "final_scratch_v2.keras", args.image, False),
    }
    out = ROOT / "artifacts" / "benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()