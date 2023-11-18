from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Chargez le modèle
#model = tf.keras.models.load_model('model_classification_images.h5')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
