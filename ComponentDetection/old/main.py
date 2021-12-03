import os
import numpy as np
import cv2
import tensorflow as tf
import json
import modelNN

# Config
DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "exported_data/"))
DATASET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "DataSet"))
DATAPIN_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "exported_data/data.json"))
IMG_SIZE = 128

def main():
    model = modelNN.getModel()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    spec = {"input1": tf.double, "input2":tf.int32},{"output1": tf.int32, "output2":tf.double}

    dataSet = tf.data.Dataset.from_generator(test, output_types=spec)

    for i in dataSet.take(1):
        print(i)

    model.fit(dataSet, epochs=20)

def test():
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
    
    x_img = np.array(x_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255

    x_hint = np.array(x_hint)
    x_hint = tf.keras.utils.to_categorical(x_hint, len(HINTS))

    y_label = np.array(y_label)
    y_label = tf.keras.utils.to_categorical(y_label, len(CATEGORIES))

    y_pin = np.array(y_pin)
    for img, hint, label, pin in zip(x_img, x_hint, y_label, y_pin):
        yield {"input1": img, "input2":hint},{"output1": label, "output2":pin}

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
    pins = [[-1,-1],[-1,-1],[-1,-1]]
    for element in pinData[categorie]:
        if(os.path.abspath(element["component_path"]).replace('\\', '/') == os.path.join(path, img)):
            count = 0
            for pinNmbr in element["pins"]:
                pins[count][0] = element["pins"][pinNmbr]["position"][0]
                pins[count][1] = element["pins"][pinNmbr]["position"][1]
                count = count + 1
    return pins
    
def Resize(image,pins):
    width, height = image.shape
    if(width > height):
        factor = IMG_SIZE/width
        image = cv2.resize(image, [round(height*factor), IMG_SIZE])
        width, height = image.shape
        delta = IMG_SIZE - height
        left, right = delta//2, delta-(delta//2)
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=255)
        for i in pins:
            if(i[0] < 0): continue
            i[0] = (((i[0]*factor)+left)/IMG_SIZE)
            i[1] = i[1]*factor/IMG_SIZE
    else:
        factor = IMG_SIZE/height
        image = cv2.resize(image, [IMG_SIZE, round(width*factor)])
        width, height = image.shape
        delta = IMG_SIZE - width
        top, bottom = delta//2, delta-(delta//2)
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)
        for i in pins:
            if(i[0] < 0): continue
            i[1] = (((i[0]*factor)+bottom)/IMG_SIZE)
            i[0] = i[1]*factor/IMG_SIZE
    pins = pins[0][0],pins[0][1],pins[1][0],pins[1][1],pins[2][0],pins[2][1]
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

if __name__ == '__main__':
    main()