PROGRAM 12: DEPLOY A DEEP LEARNING MODEL USING TENSORFLOW AND FLASK

1. REQUIREMENTS
   - Python 3.9, 3.10, or 3.11
   - Internet connection for the first MNIST dataset download

2. INSTALL PACKAGES
   Open a terminal in this project folder and run:
       python -m pip install -r requirements.txt

3. TRAIN, SAVE AND CONVERT THE MODEL
       python train_model.py

   This creates:
       saved_model/mnist_cnn.keras
       saved_model/mnist_cnn/       (TensorFlow SavedModel directory)
       saved_model/mnist_cnn.tflite

4. START THE FLASK SERVER
   Keep this terminal open and run:
       python app.py

   The API runs at:
       http://127.0.0.1:5000

5. TEST THE API
   Open a second terminal in the same project folder and run:
       python test_api.py

6. ENDPOINTS
   GET  /health
   POST /predict

   The /predict endpoint expects JSON with an image containing 784 grayscale
   pixel values arranged as a 28x28 image. Values can be 0-255 or 0-1.

NOTES
- Train the model before starting app.py.
- Do not close the Flask server terminal while running test_api.py.
- TensorFlow 2.15.1 is pinned for compatibility with the save_format='tf'
  SavedModel export used in this project.
- Training accuracy, TFLite size, prediction and inference time depend on
  the computer and training run; sample outputs are not guaranteed values.
