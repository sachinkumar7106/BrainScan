from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.models import load_model # type: ignore
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('C:/Users/Sachin/OneDrive/Desktop/MINI PROJECT/BRAIN_SCAN_AI/BrainTumor10Epochscategorical.keras')


def preprocess_image(image_path, input_size=64):
    image = cv2.imread(image_path)
    if image is not None:
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size, input_size))
        image = np.array(image)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    else:
        return None

def predict_tumor(image_path):
    image = preprocess_image(image_path)
    if image is not None:
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        return "No tumor" if class_index == 0 else "Tumor detected"
    else:
        return "Error: Image preprocessing failed."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict_tumor(file_path)
            return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
