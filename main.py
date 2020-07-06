from flask import Flask, request, make_response, redirect
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Input, Dropout, Activation
from keras import backend as X
from keras.utils import to_categorical

from keras import Model
from keras.preprocessing import image

import numpy as np

import os
from os import listdir
from os.path import isfile, join


app = Flask(__name__)

@app.route('/index')
def index():
    # Getting image from body request
    file = request.files['file']
    model = load_model('./rps.h5')
    return "{}".format(type(file))

 