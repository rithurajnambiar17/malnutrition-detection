import os
import cv2
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'INCOMING/')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods= ['POST', 'GET'])
def result():
    if request.method == 'POST':
        image = request.files['rawImage']
        oripath = app.config['UPLOAD_FOLDER'] + 'image.jpg'
        image.save(oripath)

        img = cv2.imread(oripath)
        img = img.reshape(1, 150, 150, 3)

        model = tf.keras.models.load_model('./models/model.h5')
        prediction = model.predict(img)

        if prediction[0][0] == 0:
            pred = 'No Anomaly Detected'
        else:
            pred = 'Anomaly Detected'

        return render_template('result.html', pred = pred)

app.run(debug=True)