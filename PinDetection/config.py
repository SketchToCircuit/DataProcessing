import os
from datetime import datetime 

DATAJSONPATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', "pindetection_data/data.json"))

TBDIR = "./PinDetection/logs/"
LOGDIR = os.path.join("./PinDetection/logs/", datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

MODELDIR = "./PinDetection/model/"
MODELPATH = os.path.join(MODELDIR, "t1cp.ckpt")

IMG_SIZE = 128

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