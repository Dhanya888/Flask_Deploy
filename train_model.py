import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Create output folder
os.makedirs("saved_model", exist_ok=True)

# 1. Load and preprocess MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 2. Build CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Train
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 4. Evaluate original model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Original Model Test Accuracy: {test_acc * 100:.2f}%")

# 5. Save reloadable Keras model
keras_path = "saved_model/mnist_cnn.keras"
model.save(keras_path)
print("Keras model saved:", keras_path)

# 6. Export TensorFlow SavedModel (TensorFlow 2.15.x)
saved_path = "saved_model/mnist_cnn"
if os.path.exists(saved_path):
    import shutil
    shutil.rmtree(saved_path)

model.save(saved_path, save_format="tf")
print("SavedModel exported:", saved_path)

# 7. Reload Keras model and verify accuracy
loaded_model = tf.keras.models.load_model(keras_path)
reload_loss, reload_acc = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"Reloaded Model Test Accuracy: {reload_acc * 100:.2f}%")

# 8. Convert SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
tflite_model = converter.convert()

tflite_path = "saved_model/mnist_cnn.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")

# 9. Compare original and TFLite prediction
sample = X_test[0:1]

start = time.perf_counter()
original_probs = loaded_model.predict(sample, verbose=0)[0]
original_ms = (time.perf_counter() - start) * 1000
original_prediction = int(np.argmax(original_probs))

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, sample)
start = time.perf_counter()
interpreter.invoke()
tflite_ms = (time.perf_counter() - start) * 1000

tflite_probs = interpreter.get_tensor(output_index)[0]
tflite_prediction = int(np.argmax(tflite_probs))

print("Actual label:", int(y_test[0]))
print("Original prediction:", original_prediction)
print("TFLite prediction:", tflite_prediction)
print(f"Original inference time: {original_ms:.2f} ms")
print(f"TFLite inference time: {tflite_ms:.2f} ms")
print("Training and conversion completed.")
