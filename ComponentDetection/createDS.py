import os
import numpy as np
import cv2
import random
import tensorflow as tf
import json

# Config
DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "ExportData/Data"))
DATASET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "DataSet"))
DATAPIN_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "exported_data/data.json"))

IMG_SIZE = 128

def GetDataSet():
    data = GetData()
    x_img = []
    x_hint = []
    y_label = []
    y_pin = []

    for image, hint, label, pin in data:
        x_img.append(image)
        x_hint.append(hint)
        y_label.append(label)
        #y_pin.append(pin)

    x_img = np.array(x_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    x_hint = np.array(x_hint)
    x_hint = tf.keras.utils.to_categorical(x_hint, len(HINTS))


    y_label = np.array(y_label)
    y_label = tf.keras.utils.to_categorical(y_label, len(CATEGORIES))

    #y_pin = np.array(y_pin)


    print("x_img: " , x_img.shape)
    print("x_hint: ", x_hint.shape)
    print("y_label: ", y_label.shape)
    #print("y_pin: " + y_pin.shape)

    #np.save(os.path.join(DATASET_DIR, "x_img.npy"), x_img)
    #np.save(os.path.join(DATASET_DIR, "x_hint.npy"), x_hint)
    #np.save(os.path.join(DATASET_DIR, "Y.npy"), y_label)


def GetData():
    pinFile = open(DATAPIN_FILE)
    pinData = json.load(pinFile)
    training_data = []
    for categorie in CATEGORIES:
        path = os.path.join(DATA_DIR, categorie)
        class_num = CATEGORIES.index(categorie)
        hint = HINTS.index(COMPONENTS[categorie])
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            img_array = image_resize(img_array, width=IMG_SIZE, height=IMG_SIZE)
            cv2.imshow('image', img_array)
            training_data.append([img_array, hint, class_num, None])
    random.shuffle(training_data)
    return training_data

def GetPins():
    print()

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

CATEGORIES = ["R","C","L","LED","D","POT","S1","S2","BTN1","BTN2","V_V","V_H","A_H","A_V","U1","U2","I1","I2","U3","BAT","U_AC_H",
              "U_AC_V","SPK","MIC","LMP","M","S3","F","OPV","NPN","PNP","GND","GND_F","GND_C","D_Z","D_S","MFET_N_E","MFET_P_E","MFET_N_D","L2",
              "MFET_P_D","JFET_N","JFET_P","C_P","PIN","M_V"]

HINTS = ["circle", "battery", "button", "capacitor", "diode", "resistor", "ground", "transistor", "coil", "microphone", "opv", "pin", "potentiometer", "speaker", "l2"]

COMPONENTS = {
    'A_H': 'circle',
    'A_V': 'circle',
    'BAT': 'battery',
    'BTN1': 'button',
    'BTN2': 'button',
    'C': 'capacitor',
    'C_P': 'capacitor',
    'D': 'diode',
    'D_S': 'diode',
    'D_Z': 'diode',
    'F': 'resistor',
    'GND': 'ground',
    'GND_C': 'ground',
    'GND_F': 'ground',
    'I1': 'circle',
    'I2': 'circle',
    'JFET_N': 'transistor',
    'JFET_P': 'transistor',
    'L': 'coil',
    'LED': 'diode',
    'LMP': 'circle',
    'M': 'circle',
    'M_V': 'circle',
    'MFET_N_D': 'transistor',
    'MFET_N_E': 'transistor',
    'MFET_P_D': 'transistor',
    'MFET_P_E': 'transistor',
    'MIC': 'microphone',
    'NPN': 'transistor',
    'OPV': 'opv',
    'PIN': 'pin',
    'PNP': 'transistor',
    'POT': 'potentiometer',
    'R': 'resistor',
    'S1': 'button',
    'S2': 'button',
    'S3': 'button',
    'SPK': 'speaker',
    'U1': 'circle',
    'U2': 'circle',
    'U3': 'circle',
    'U_AC_H': 'circle',
    'U_AC_V': 'circle',
    'V_H': 'circle',
    'V_V': 'circle',
    'L2' : 'l2'
}

GetDataSet()