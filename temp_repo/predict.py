import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_food_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map.pkl")

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}


# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path, threshold=0.6):

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)

    class_index = np.argmax(preds)
    confidence = float(np.max(preds))
    class_name = inv_label_map[class_index]

    # Confidence check
    if confidence < threshold:
        return "Not sure (low confidence)", confidence

    return class_name, confidence


# ---------------- TEST RUN ----------------
if __name__ == "__main__":
    test_image = "test.jpg"  # change path if needed
    name, conf = predict_image(test_image)

    print(f"Prediction: {name}")
    print(f"Confidence: {conf:.2f}")
