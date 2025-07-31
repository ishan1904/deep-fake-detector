import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

#app initialisation

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]= "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#loading model
MODEL_PATH = os.path.join("model", "deepfake_detector.keras")
model = load_model(MODEL_PATH)

IMG_SIZE = 128

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)

            # Preprocess the image
            img = load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Make prediction
            result = model.predict(img)[0][0]
            
            # Interpret the prediction
            # Interpret the prediction
            label = "Fake" if result < 0.5 else "Real"
            confidence = 1 - result if label == "Fake" else result
            confidence_percent = round(confidence * 100, 2)
            prediction = f"{label} face with {confidence_percent}% confidence."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
