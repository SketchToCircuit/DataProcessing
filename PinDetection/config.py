import os
from datetime import datetime 

DATAJSONPATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "pindetection_data/data.json"))

TBDIR = "./PinDetection/logs/"
LOGDIR = os.path.join("./PinDetection/logs/", datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

TRAINMODELDIR = "./PinDetection/testModel/"
TRAINMODELPATH = os.path.join(TRAINMODELDIR, "t1cp.ckpt")

IMG_SIZE = 64

FINE_TO_COARSE_PATH = './ObjectDetection/fine_to_coarse_labels.txt'
LABEL_CONVERT_DICT = {}
with open(FINE_TO_COARSE_PATH, 'r') as f:
    for l in f:
        values = l.split(',')
        LABEL_CONVERT_DICT[values[0]] = (values[1], int(values[2]))

# CATEGORIES = ["R","C","L","LED","D","POT","S1","S2","BTN1","BTN2","V_V","V_H","A_H","A_V","U1","U2","I1","I2","U3","BAT","U_AC_H",
#               "U_AC_V","SPK","MIC","LMP","M","S3","F","OPV","NPN","PNP","GND","GND_F","GND_C","D_Z","D_S","MFET_N_E","MFET_P_E","MFET_N_D","L2",
#               "MFET_P_D","JFET_N","JFET_P","C_P","PIN","M_V"]