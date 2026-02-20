from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)

MODEL_PATH = "healthy_vs_rotten.h5"
model = tf.keras.models.load_model(r"C:\\Users\\abhim\\vs code\\py\Smart Sorting Transfer Learning for Identifying Rotten Fruits and Vegetables\\healthy_vs_rotten.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Invalid Image"

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Rotten ❌"
    else:
        return "Healthy / Fresh ✅"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/portfolio")
def portfolio():
    return render_template("portfolio-details.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")


@app.route("/blog-single")
def blog_single():
    return render_template("blog-single.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("portfolio-details.html", prediction="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("portfolio-details.html", prediction="No file selected")

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result = predict_image(filepath)

    return render_template("portfolio-details.html", prediction=result, image_path=filepath)


if __name__ == "__main__":
    app.run(debug=True)
