from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('fv.h5')

# Define classes for fruits/vegetables
classes = ['apple', 'banana', 'carrot', ...]  # Define your classes here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Perform inference
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]
    # You may need to adjust how you extract and format additional information like calories based on your model's output
    
    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
