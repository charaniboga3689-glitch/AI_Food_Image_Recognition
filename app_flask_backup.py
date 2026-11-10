import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, url_for, redirect
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "model" / "food_model.keras"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"
HISTORY_PATH = BASE_DIR / "model" / "prediction_history.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_class_names = None

def read_history():
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []

def save_history(items):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(items[:100], indent=2), encoding="utf-8")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
    global _model, _class_names
    if _model is None:
        if not MODEL_PATH.exists() or not CLASS_PATH.exists():
            raise FileNotFoundError(
                "The trained model is not installed yet. Follow README.md to download "
                "Food-101 and run python train.py."
            )
        _model = load_model(MODEL_PATH)
        _class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    return _model, _class_names

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    image_url = None

    if request.method == "POST":
        uploaded = request.files.get("food_image")
        if not uploaded or uploaded.filename == "":
            error = "Please choose a food image first."
        elif not allowed_file(uploaded.filename):
            error = "Please upload a JPG, JPEG, PNG, or WEBP image."
        else:
            try:
                image = Image.open(uploaded.stream).convert("RGB")
                image.thumbnail((1600, 1600))
                filename = f"{uuid.uuid4().hex}.jpg"
                saved_path = UPLOAD_DIR / filename
                image.save(saved_path, format="JPEG", quality=92)

                model, class_names = get_model()
                resized = image.resize(IMAGE_SIZE)
                array = preprocess_input(np.asarray(resized, dtype=np.float32))
                predictions = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
                top_indices = np.argsort(predictions)[::-1][:3]

                result = {
                    "name": class_names[int(top_indices[0])].replace("_", " ").title(),
                    "confidence": round(float(predictions[int(top_indices[0])]) * 100, 2),
                    "alternatives": [
                        {"name": class_names[int(i)].replace("_", " ").title(),
                         "confidence": round(float(predictions[int(i)]) * 100, 2)}
                        for i in top_indices[1:]
                    ]
                }
                image_url = url_for("static", filename=f"uploads/{filename}")
                history = read_history()
                history.insert(0, {
                    "food": result["name"],
                    "confidence": result["confidence"],
                    "image": f"uploads/{filename}",
                    "date": datetime.now().strftime("%d %b %Y, %I:%M %p")
                })
                save_history(history)
            except FileNotFoundError as exc:
                error = str(exc)
            except Exception as exc:
                error = f"Could not process this image. Please try another image. Details: {exc}"

    return render_template("index.html", error=error, result=result, image_url=image_url)

@app.route("/dashboard")
def dashboard():
    history = read_history()
    total = len(history)
    average = round(sum(x.get("confidence", 0) for x in history) / total, 1) if total else 0
    popular = {}
    for item in history:
        popular[item.get("food", "Unknown")] = popular.get(item.get("food", "Unknown"), 0) + 1
    top_foods = sorted(popular.items(), key=lambda x: x[1], reverse=True)[:5]
    return render_template("dashboard.html", history=history[:6], total=total,
                           average=average, top_foods=top_foods)

@app.route("/history")
def history_page():
    return render_template("history.html", history=read_history())

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", error="Image is too large. Please upload an image smaller than 8 MB."), 413

if __name__ == "__main__":
    app.run(debug=True)
