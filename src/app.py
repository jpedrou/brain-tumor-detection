import os
import cv2
import warnings
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")

from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# =====================================================
# Create Flask app
# =====================================================

app = Flask(__name__)

model = load_model("../model.h5")
print("Model loaded. Check http://127.0.0.1:5000/")

# =====================================================
# util functions
# =====================================================


def get_result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, "RGB")
    image = image.resize((64, 64))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)

    return result


def predict_result(y_pred):
    if y_pred > 0.5:
        return "tumor detected"
    else:
        return "No tumor detected"


# =====================================================
# Routes
# =====================================================


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)
        value = get_result(file_path)
        result = predict_result(value)
        return {"prediction": result}
    return None


if __name__ == "__main__":
    app.run(debug=True)
