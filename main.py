from flask import Flask, request, make_response, redirect
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Input, Dropout, Activation
from keras import backend as X
from keras.utils import to_categorical

from keras import Model
import numpy as np

import os
from os import listdir
from os.path import isfile, join

from PIL import Image
import io

import random

app = Flask(__name__)

@app.route('/index', methods=["POST"])
def index():
    
    # Getting image from body request
    file = request.files['file']
    
    # load model
    model = load_model('./rps-2.h5')
    
    #compile model
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    # proccessing request body image

    img = Image.open(file)
    rgb_im = img.convert('RGB')
    path = "./img/{}.jpg".format(random.randint(1,100000))
    rgb_im.save(path)
    img = load_img(path, target_size=(150, 150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    return "{}".format(classes)

 