import requests
import tensorflow as tf

# Use an MNIST test image as the API input
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
sample_image = X_test[0]

# Test health endpoint
health_response = requests.get("http://127.0.0.1:5000/health", timeout=10)
print("Health response:", health_response.json())

# Send image to prediction endpoint
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"image": sample_image.tolist()},
    timeout=30
)

print("HTTP status:", response.status_code)
result = response.json()
print("API response:", result)

if response.ok:
    print("Actual digit:", int(y_test[0]))
    print("Predicted digit:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Inference time:", result["inference_ms"], "ms")
