import hyper as hp
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import json
import cv2
import os
import base64
from PIL import Image
import io

global model

print("upper")
__model = None
#path của file
file_path=os.path.dirname(os.path.abspath('__file__'))+os.sep
#path vào artifacts
artifacts_path=os.path.join(file_path,"artifacts")
#haar path
haar_path=os.path.join(artifacts_path,"haarcascade_frontalface_default.xml")
#ferjson path
ferjson_path=os.path.join(artifacts_path,"fer.json")
#fer h5 path
ferh5_path=os.path.join(artifacts_path,"fer.h5")
#croping face
face_classifier = cv2.CascadeClassifier(haar_path)
#class dict path
server_path=os.path.join(file_path,"server")
class_dict_path=os.path.join(server_path,"class_dictionary.json")
test_img_path=os.path.join(server_path,"suprise.jpg")
__class_name_to_number = {}
__class_number_to_name = {}


def classify_image(img_base64_data,path=None):
    imgs = get_cropped_image_if_2_eyes(path, img_base64_data)

    result = []
    for img in imgs:
        print(__model.predict(img))
        result.append({
            'class': class_number_to_name(np.argmax(__model.predict(img))),
            'class_probability': np.round(__model.predict(img)[0]*100).tolist(),
            'class_dictionary': __class_name_to_number
        })
    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(class_dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    __model = model_from_json(open(ferjson_path, "r").read())
    __model.load_weights(ferh5_path)
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray=cv2.resize(roi_gray,(hp.IMAGE_HEIGHT,hp.IMAGE_WIDTH)) 
        img = image.img_to_array(roi_gray)
        img = np.expand_dims(img,axis=0)
        img /= 255
        cropped_faces.append(img)

    return cropped_faces

if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(None,test_img_path))  

