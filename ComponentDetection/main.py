from enum import Enum
import os
import tensorflow as tf
import numpy as np

# Config
data_dir = "exported_data"
#img_height = 128
batch_size = 1000
seed = 423


# 45
#oneHot = np.array()

def main():
    data = loadData()
    print(data)


def loadData():
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    allDirList = os.listdir(data_dir)
    dirList = list()
    for i in allDirList:
        if(i == "data.json"):
            continue
        if(not i.endswith('_label')):
            dirList.append(i)
    allFiles = list()
    for i in dirList:
        if(i == "data.json"):
            continue
        allFiles.append(os.listdir(os.path.join(data_dir, i)))
    return dirList


# def customGenerator(data):
#    for i in data:


Components = {
    'circle': 'A_H',
    'circle': 'A_V',
    'battery': 'BAT',
    'button': 'BTN1',
    'button': 'BTN2',
    'capacitor': 'C',
    'capacitor': 'C_P',
    'diode': 'D',
    'diode': 'D_S',
    'diode': 'D_Z',
    'resistor': 'F',
    'ground': 'GND',
    'ground': 'GND_C',
    'ground': 'GND_F',
    'circle': 'I1',
    'circle': 'I2',
    'transistor': 'JFET_N',
    'transistor': 'JFET_P',
    'coil': 'L',
    'diode': 'LED',
    'circle': 'LMP',
    'circle': 'M',
    'circle': 'M_V',
    'transistor': 'MFET_N_D',
    'transistor': 'MFET_N_E',
    'transistor': 'MFET_P_D',
    'transistor': 'MFET_P_E',
    'microphone': 'MIC',
    'transistor': 'NPN',
    'opv': 'OPV',
    'pin': 'PIN',
    'transistor': 'PNP',
    'potentiometer': 'POT',
    'resistor': 'R',
    'button': 'S1',
    'button': 'S2',
    'button': 'S3',
    'speaker': 'SPK',
    'circle': 'U1',
    'circle': 'U2',
    'circle': 'U3',
    'circle': 'U_AC_H',
    'circle': 'U_AC_V',
    'circle': 'V_H',
    'circle': 'V_V'
}

if __name__ == '__main__':
    main()
