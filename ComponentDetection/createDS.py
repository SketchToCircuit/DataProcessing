import os
import numpy as np
import cv2
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# Config
data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "ExportData/Data"))
dataset_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "DataSet"))
IMG_SIZE = 128

def main():
    data = createDS()
    X = []
    X_HINT = []
    Y = []
    for features, hint, label in data:
        X.append(features)
        X_HINT.append(hint)
        Y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    Y = np.array(Y)
    Y = tf.keras.utils.to_categorical(Y, len(CATEGORIES))
#    Y = tf.convert_to_tensor(Y)

    X_HINT = np.array(X_HINT)
    X_HINT = tf.keras.utils.to_categorical(X_HINT, len(HINTS))
#    X_HINT = tf.convert_to_tensor(X_HINT)

    np.save(os.path.join(dataset_dir, "X.npy"), X)
    np.save(os.path.join(dataset_dir, "X_HINT.npy"), X_HINT)
    np.save(os.path.join(dataset_dir, "Y.npy"), Y)


def createDS():
    training_data = []
    for categorie in CATEGORIES:
        path = os.path.join(data_dir, categorie)
        class_num = CATEGORIES.index(categorie)
        hint = HINTS.index(COMPONENTS[categorie])
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([img_array, hint, class_num])
    random.shuffle(training_data)
    return training_data

#x_daten Lade Bilder -> [Categorien[Bilder[Bildaten, onehot]]]
#y_daten Lade Bilder -> [Categorien[Bilder[dirName]]]
# def customGenerator(data):
#    for i in data:

CATEGORIES = ["R","C","L","LED","D","POT","S1","S2","BTN1","BTN2","V_V","V_H","A_H","A_V","U1","U2","I1","I2","U3","BAT","U_AC_H",
              "U_AC_V","SPK","MIC","LMP","M","S3","F","OPV","NPN","PNP","GND","GND_F","GND_C","D_Z","D_S","MFET_N_E","MFET_P_E","MFET_N_D",
              "MFET_P_D","JFET_N","JFET_P","C_P","PIN","M_V"]

HINTS = ["circle", "battery", "button", "capacitor", "diode", "resistor", "ground", "transistor", "coil", "microphone", "opv", "pin", "potentiometer", "speaker"]

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
    'V_V': 'circle'#,
#    'L2' : 'L2'
}

if __name__ == '__main__':
    main()
