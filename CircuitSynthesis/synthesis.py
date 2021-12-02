from dataclasses import dataclass
from typing import Tuple, List

from cv2 import waitKey
import Tools.PinDetection.pindetection as pd
from Tools.squigglylines import * 
from Tools.autoroute import *

import random
import numpy as np

DROP_LINE_PERCENTAGE = 0.2
PART_COUNT_MU = 20 #mü is the amount of average parts
PART_COUNT_SIGMA = 5 #sigma is standart deviation
MAX_GRIDSIZE_OFFSET = 25


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

def augment(component: Component):
    #rotate
    #flip
    allowed = ["C", "D", "C_P", "D_S", "D_Z", "L", "LED", "L", "R"]
    if component.type in allowed:
           


def main():
    components = pd.import_components('./exported_data/data.json')
    
    partamount = int(np.random.normal(PART_COUNT_MU, PART_COUNT_SIGMA, 1))
    partcount = 0
    if partamount < 3:
        partamount = 3

    gridsize = 150
    pos = ()
    compList: List[CirCmp] = []
    conList :List[Tuple[CirCmp, Pin, CirCmp, Pin]] = []
    unsatisfiedList = [] # sollte durch statisches array ersretzt werden
    #randomly define colums and rows
    rancols = random.randint(3, 7)
    ranrows = math.ceil(partamount / rancols)
    for i in range(rancols):
        for j in range(ranrows):
            componentSize = int(random.randint(64,128))
            pos = (
            j * gridsize + random.randint(-MAX_GRIDSIZE_OFFSET, MAX_GRIDSIZE_OFFSET),#X
            i * gridsize + random.randint(-MAX_GRIDSIZE_OFFSET, MAX_GRIDSIZE_OFFSET))#Y

            random_type = random.sample(components.keys(), k=1)[0]
            cmp = random.sample(components[random_type], k=1)[0]

            cmp = cmp.load()



            cmp.scale(componentSize / np.max(cmp.component_img.shape))
            

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

    conList = random.sample(conList, int(len(conList) * ( 1 - DROP_LINE_PERCENTAGE)))

    cv2.imshow("reset",draw_routed_circuit(route(compList, conList),True))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()