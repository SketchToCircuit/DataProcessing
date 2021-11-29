import os
import numpy as np
import cv2
import tensorflow as tf
import json

# Config
DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "exported_data/"))
DATASET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "DataSet"))
DATAPIN_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "exported_data/data.json"))
IMG_SIZE = 128

def GetDataSet():

    specs =(
        tf.TensorSpec(shape=(IMG_SIZE,IMG_SIZE,1), dtype=tf.double),
        tf.TensorSpec(shape=(16), dtype=tf.int32),
        tf.TensorSpec(shape=(46), dtype=tf.int32),
        tf.TensorSpec(shape=(6), dtype=tf.double))
    dataSet = tf.data.Dataset.from_generator(gen, output_signature=specs)
    return dataSet


def gen():
    data = GetData()
    x_img = []
    x_hint = []
    y_label = []
    y_pin = []

    for image, hint, label, pin in data:
        x_img.append(image)
        x_hint.append(hint)
        y_label.append(label)
        y_pin.append(pin)
    
    x_img = np.array(x_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    x_hint = np.array(x_hint)
    x_hint = tf.keras.utils.to_categorical(x_hint, len(HINTS))

    y_label = np.array(y_label)
    y_label = tf.keras.utils.to_categorical(y_label, len(CATEGORIES))

    y_pin = np.array(y_pin)

    for x_1, x_2, y_1, y_2 in zip(x_hint, x_img, y_label, y_pin):
        yield {'input1': x_1, 'input2': x_2}, {'output1':y_1, 'output2':y_2}


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
            pinLoc = GetPinLoc(pinData,categorie ,img, path)
            img_array,pinLoc = Resize(img_array, pinLoc)

            training_data.append([img_array, hint, class_num, pinLoc])
    return training_data

def GetPinLoc(pinData, categorie, img, path):
    for element in pinData[categorie]:
        if(os.path.abspath(element["component_path"]).replace('\\', '/') == os.path.join(path, img)):
            pins = []
            if(categorie == "NPN" or categorie == "PNP"):
                pins.append(element["pins"]["base"]["position"])
                pins.append(element["pins"]["collector"]["position"])
                pins.append(element["pins"]["emitter"]["position"])
            elif(categorie == "MFET_N_E" or categorie == "MFET_P_E" or categorie == "MFET_N_D" or categorie == "MFET_P_D" or categorie == "JFET_N" or categorie == "JFET_P"):
                pins.append(element["pins"]["gate"]["position"])
                pins.append(element["pins"]["drain"]["position"])
                pins.append(element["pins"]["source"]["position"])
            elif(categorie == "S3" or categorie == "OPV"):
                pins.append(element["pins"]["+"]["position"])
                pins.append(element["pins"]["-"]["position"])
                pins.append(element["pins"]["out"]["position"])
            else:
                if(len(element["pins"]) == 1):
                    pins.append(element["pins"]["1"]["position"])
                    pins.append([None, None])
                    pins.append([None, None])
                elif(len(element["pins"]) == 2):
                    pins.append(element["pins"]["1"]["position"])
                    pins.append(element["pins"]["2"]["position"])
                    pins.append([None, None])
                elif(len(element["pins"]) == 3):
                    pins.append(element["pins"]["1"]["position"])
                    pins.append(element["pins"]["2"]["position"])
                    pins.append(element["pins"]["3"]["position"])
            if(len(pins) == 3):
                return pins


    #Temp!!
    pins = []
    pins.append([None, None])
    pins.append([None, None])
    pins.append([None, None])
    return pins


def Resize(image, pins):
    width, height = image.shape
    if(width > height):
        
        factor = IMG_SIZE/width
        
        image = cv2.resize(image, [round(height*factor), IMG_SIZE])
        width, height = image.shape
        delta = IMG_SIZE - height
        left, right = delta//2, delta-(delta//2)
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=255)
        if(pins[0][0] != None and pins[0][1] != None):
            pins[0][0] = pins[0][0]*factor
            pins[0][1] = pins[0][1]*factor/IMG_SIZE
            pins[0][0] = (pins[0][0]+left)/IMG_SIZE
        if(pins[1][0] != None and pins[1][1] != None):
            pins[1][0] = pins[1][0]*factor
            pins[1][1] = pins[1][1]*factor/IMG_SIZE
            pins[1][0] = (pins[1][0]+left)/IMG_SIZE
        if(pins[2][0] != None and pins[2][1] != None):
            pins[2][0] = pins[2][0]*factor
            pins[2][1] = pins[2][1]*factor/IMG_SIZE
            pins[2][0] = (pins[2][0]+left)/IMG_SIZE
    else:
        factor = IMG_SIZE/height
        image = cv2.resize(image, [IMG_SIZE, round(width*factor)])
        width, height = image.shape
        delta = IMG_SIZE - width
        top, bottom = delta//2, delta-(delta//2)
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)
        if(pins[0][0] != None and pins[0][1] != None):
            pins[0][0] = pins[0][0]*factor/IMG_SIZE
            pins[0][1] = pins[0][1]*factor
            pins[0][1] = (pins[0][1] + bottom)/IMG_SIZE
        if(pins[1][0] != None and pins[1][1] != None):
            pins[1][0] = pins[1][0]*factor/IMG_SIZE
            pins[1][1] = pins[1][1]*factor
            pins[1][1] = (pins[1][1] + bottom)/IMG_SIZE
        if(pins[2][0] != None and pins[2][1] != None):
            pins[2][0] = pins[2][0]*factor/IMG_SIZE
            pins[2][1] = pins[2][1]*factor
            pins[2][1] = (pins[2][1] + bottom)/IMG_SIZE
    pins = [pins[0][0],pins[0][1],pins[1][0],pins[1][1],pins[2][0],pins[2][1]]
    return image, pins



CATEGORIES = ["R","C","L","LED","D","POT","S1","S2","BTN1","BTN2","V_V","V_H","A_H","A_V","U1","U2","I1","I2","U3","BAT","U_AC_H",
              "U_AC_V","SPK","MIC","LMP","M","S3","F","OPV","NPN","PNP","GND","GND_F","GND_C","D_Z","D_S","MFET_N_E","MFET_P_E","MFET_N_D","L2",
              "MFET_P_D","JFET_N","JFET_P","C_P","PIN","M_V"]

HINTS = ["circle", "battery", "button", "capacitor", "diode", "resistor", "ground", "transistor", "coil", "microphone", "opv", "pin", "potentiometer", "s3" ,"speaker", "l2"]

COMPONENTS = { 'A_H': 'circle', 'A_V': 'circle', 'BAT': 'battery','BTN1': 'button','BTN2': 'button','C': 'capacitor','C_P': 'capacitor',
'D': 'diode','D_S': 'diode','D_Z': 'diode','F': 'resistor','GND': 'ground','GND_C': 'ground','GND_F': 'ground','I1': 'circle','I2': 'circle',
'JFET_N': 'transistor','JFET_P': 'transistor','L': 'coil','LED': 'diode','LMP': 'circle','M': 'circle','M_V': 'circle','MFET_N_D': 'transistor',
'MFET_N_E': 'transistor','MFET_P_D': 'transistor','MFET_P_E': 'transistor','MIC': 'microphone','NPN': 'transistor','OPV': 'opv','PIN': 'pin',
'PNP': 'transistor','POT': 'potentiometer','R': 'resistor','S1': 'button','S2': 'button','S3': 's3','SPK': 'speaker','U1': 'circle','U2': 'circle',
'U3': 'battery','U_AC_H': 'circle','U_AC_V': 'circle','V_H': 'circle','V_V': 'circle','L2' : 'l2'
}

GetDataSet()