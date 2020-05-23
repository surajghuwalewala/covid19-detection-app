import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model, model_from_json
from werkzeug.utils import secure_filename
import numpy as np
import datetime



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'

# model = load_model(os.path.join("model","denseNet169_combined.h5"))

json_file = open(os.path.join("model","denseNet169_comb.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join("model","denseNet169_comb_weights.h5"))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def minMaxNorm(img):
    return (img - np.min(img))/(np.max(img) - np.min(img))


def predict(file):
    img  = load_img(file, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = minMaxNorm(img)
    # img = np.expand_dims(img, axis=0)
    img = img.reshape((1,224,224,3))
    probs = model.predict(img)[0]
    output = 'Probability of Covid-19 infection:   {:.2f}%'.format(probs[0]*100)#, 'Dog': probs[1]}
    return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], datetime.datetime.now().strftime("%H-%M-%S")+filename
, )
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
