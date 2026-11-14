from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import time
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("saved_model", "mnist_cnn.keras")
model = tf.keras.models.load_model(MODEL_PATH)
print("MNIST CNN model loaded successfully.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "mnist_cnn"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({"error": "Send JSON containing an 'image' field."}), 400

    try:
        image = np.asarray(data["image"], dtype=np.float32)

        if image.size != 784:
            return jsonify({"error": "Image must contain exactly 784 pixel values (28x28)."}), 400

        image = image.reshape(1, 28, 28, 1)

        if not np.all(np.isfinite(image)):
            return jsonify({"error": "Image contains invalid pixel values."}), 400

        # Accept either 0-255 grayscale pixels or already-normalized 0-1 pixels.
        if image.max() > 1.0:
            image = image / 255.0

        if image.min() < 0.0 or image.max() > 1.0:
            return jsonify({"error": "Pixel values must be between 0 and 255."}), 400

        start = time.perf_counter()
        probabilities = model.predict(image, verbose=0)[0]
        inference_ms = (time.perf_counter() - start) * 1000

        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "all_probs": [round(float(p), 4) for p in probabilities],
            "inference_ms": round(inference_ms, 2)
        })

    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
