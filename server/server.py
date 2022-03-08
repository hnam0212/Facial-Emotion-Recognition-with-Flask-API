from PIL import Image,ImageOps
import numpy as np
from tensorflow.python import util
import hyper as hp
from flask import Flask, app, request,jsonify
import flask
import json
import io
import utils

import cv2



#khởi tạo flask
app = Flask(__name__)

#khai báo route default
@app.route("/",methods=['GET'])
def hello():
    return "welcome to my api"

@app.route("/predict",methods=['GET', 'POST'])
#predict function
def predict():
    if request.files.get("image"):
        image_data=request.files["image"].read()
    else:    
        image_data = request.form['image_data']
    print(image_data)
    response = jsonify(utils.classify_image(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server")
    utils.load_saved_artifacts()
    app.run(port=5000)