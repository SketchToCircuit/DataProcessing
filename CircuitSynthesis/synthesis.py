from typing import Dict, Tuple, List

from Tools.export_tfrecords import export_circuits, export_label_map
import Tools.PinDetection.pindetection as pd
from Tools.squigglylines import * 
from Tools.autoroute import *

import random
import numpy as np

DROP_LINE_PERCENTAGE = 0.4
PART_COUNT_MU = 20 #mü is the amount of average parts
PART_COUNT_SIGMA = 5 #sigma is standart deviation

MAX_GRIDSIZE_OFFSET = 25
GRIDSIZE = 170

NUM_FILES = 20
CIRCUITS_PER_FILE = 1000
VALIDATION_NUM = 200
VAL_SRC_SPLIT = 0.1

DEBUG = False

def _bridgecircuit(compList: List[Tuple], conList: List[Tuple], components, pos):
    cmps = []

    for i in range(4):
        allowed = ["C", "D", "C_P", "D_S", "D_Z", "L", "LED", "L", "R"]
        while random_type not in allowed:
            random_type = random.sample(components.keys(), k=1)[0]
        cmps[i] = random.sample(components[random_type], k=1)[0]
        cmps[i] = cmps.load()
    
    cmps: List[Component]
    componentSize = int(random.randint(64,128))
    #fpos = first position = the postition of combined bounding box
    
    randangle = random.randint(-2,2)

    componentSize = int(random.randint(64,128))

    cmps[0].scale(componentSize / np.max(cmps[0].component_img.shape))
    cmps[0].rotate(45 + randangle)

    allowed = ["C", "D", "C_P", "D_S", "D_Z", "LED"]

    for i in range(4):
        if cmps[i].type() in allowed:
            cmps[i].scale(int(componentSize * 0.5 / np.max(cmps[0].component_img.shape)))
        else:
            cmps[i + 1].scale(int(componentSize / np.max(cmps[0].component_img.shape)))
        cmps[i].rotate(45 + randangle + i * 90)
    
    #components get counted counter  clockwise
    compList.append(CirCmp(cmps[0].type(), cmps[0], pos))
    compList.append(CirCmp(cmps[1].type(), cmps[1], (pos[0], pos[1] + cmps[0].component_img.shape[0])))
    compList.append(CirCmp(cmps[2].type(), cmps[2], (pos[0] + np.max(cmps[0].component_img[1], cmps[1].component_img[1]), pos[1] + cmps[0].component_img.shape[0])))
    compList.append(CirCmp(cmps[3].type(), cmps[3], (pos[0] + np.max(cmps[0].component_img[1], cmps[1].component_img[1]), pos[1])))

    compList.append(CirCmp("knot", None, None))
    #compute knots

def _augment(component: Component):

    if random.random() < 0.5:
        component.flip(vert=True)

    # some types have their own vertical version -> don't rotate 90°
    both_versions = ["A_H", "A_V", "U_AC_H", "U_AC_V", "V_H", "V_V", "M", "M_V"]
    # some types can be diagonal (45°)
    diagonal_allowed = ["C", "C_P", "D", "D_S", "D_Z", "L", "L2", "R"]

    angle = 0.0

    if component.type in both_versions:
        angle = random.choice([0.0, 180.0])
    elif component.type in diagonal_allowed:
        # only 10% are diagonal
        if random.random() < 0.1:
            angle = random.choice([45.0, -45.0, 135.0, -135.0])
        else:
            angle = random.choice([0.0, 90.0, -90.0, 180.0])
    else:
        angle = random.choice([0.0, 90.0, -90.0, 180.0])
    
    angle += np.random.normal(0, 10)

    component.rotate(angle)

def _create_circuit(components: Dict[str, pd.UnloadedComponent], validation=False):
    partamount = int(np.random.normal(PART_COUNT_MU, PART_COUNT_SIGMA, 1))
    if partamount < 3:
        partamount = 3

    pos = ()
    compList: List[CirCmp] = []
    conList: List[Tuple[CirCmp, Pin, CirCmp, Pin]] = []
    unsatisfiedList = [] # sollte durch statisches array ersretzt werden
    #randomly define colums and rows
    rancols = random.randint(3, 7)
    ranrows = math.ceil(partamount / rancols)

    # If both versions are available -> make them half as likely to get choosen
    both_versions = ["A_H", "A_V", "U_AC_H", "U_AC_V", "V_H", "V_V", "M", "M_V"]
    weights = [0.5 if t in both_versions else 1.0 for t in components.keys()]

    for i in range(rancols):
        for j in range(ranrows):
            pos = (
            j * GRIDSIZE + random.randint(-MAX_GRIDSIZE_OFFSET, MAX_GRIDSIZE_OFFSET),#X
            i * GRIDSIZE + random.randint(-MAX_GRIDSIZE_OFFSET, MAX_GRIDSIZE_OFFSET))#Y
            
            random_type = random.choices([*components.keys()], weights=weights, k=1)[0]
            num_val = int(len(components[random_type]) * VAL_SRC_SPLIT)

            if validation:
                rand_idx = random.randint(0, num_val - 1)
            else:
                rand_idx = random.randint(num_val - 1, len(components[random_type]) - 1)
            
            cmp = components[random_type][rand_idx]
            
            #Loaded components are enabled to Edit
            cmp = cmp.load()

            #make some components bigger[by now only the OPV]
            bigger = ["OPV"]
            componentSize = int(random.randint(64,128))
            if cmp.type in bigger:
                cmp.scale((componentSize + 40) / np.max(cmp.component_img.shape))
            else:
                cmp.scale(componentSize / np.max(cmp.component_img.shape))

            _augment(cmp)

            newEntry = CirCmp(random_type, cmp, pos)
            compList.append(newEntry)

    for i in range(partamount):
        if len([*compList[i - 1].cmp.pins.values()]) == 3:
            unsatisfiedList.append((i - 1, 2))#third pin
        elif len([*compList[i - 1].cmp.pins.values()]) == 1:
             unsatisfiedList.append((i - 1, 0))#thirst pin
        else:
            conList.append((
               compList[i - 1],
               [*compList[i - 1].cmp.pins.values()][1],
               compList[i],
               [*compList[i].cmp.pins.values()][0]
            ))

    for i in range(len(unsatisfiedList)):
        conList.append((
              compList[unsatisfiedList[i - 1][0]],
              [*compList[unsatisfiedList[i - 1][0]].cmp.pins.values()][unsatisfiedList[i - 1][1]],
              compList[unsatisfiedList[i][0]],
              [*compList[unsatisfiedList[i][0]].cmp.pins.values()][unsatisfiedList[i][1]]
            ))

    for i in range(int(len(compList) * 0.5)):
        compList.append(CirCmp("knot", None, None))
        conList.append((compList[-1], None, compList[i], random.choice([*compList[i].cmp.pins.values()])))
        conList.append((compList[-1], None, compList[i * 2], random.choice([*compList[i * 2].cmp.pins.values()])))

    conList = random.sample(conList, int(len(conList) * ( 1 - DROP_LINE_PERCENTAGE)))

    return(route(compList, conList))

if __name__ == '__main__':
    export_label_map('./DataProcessing/ObjectDetection/data/label_map.pbtxt', './DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')

    components = pd.import_components('./DataProcessing/pindetection_data/data.json')

    if DEBUG:
        circ = _create_circuit(components)
        cv2.imshow('', draw_routed_circuit(circ, labels=True))
        cv2.waitKey(0)
        exit()

    val_cirucits: List[RoutedCircuit] = [None] * VALIDATION_NUM
    for i in range(VALIDATION_NUM):
        val_cirucits[i] = _create_circuit(components, validation=True)

        if i % 100 == 0:
            print(f'val:{i}')
    
    export_circuits(val_cirucits, f'./DataProcessing/ObjectDetection/data/val.tfrecord', './DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')

    for f in range(NUM_FILES):
        cirucits: List[RoutedCircuit] = [None] * CIRCUITS_PER_FILE
        
        for i in range(CIRCUITS_PER_FILE):
            cirucits[i] = _create_circuit(components)

            if i % 200 == 0:
                print(f'{f}:{i}')

        export_circuits(cirucits, f'./DataProcessing/ObjectDetection/data/train-{f}.tfrecord', './DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')