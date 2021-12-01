from dataclasses import dataclass
from typing import Tuple, List

from cv2 import waitKey
import Tools.PinDetection.pindetection as pd
from Tools.squigglylines import * 
from Tools.autoroute import *

import random
import numpy as np

def bridgecircuit(compList: List[Tuple], conList: List[Tuple], components, pos):
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

def starcircuit(compList: List[Tuple], conList: List[Tuple]):
    pass
def deltacircuit(compList: List[Tuple], conList: List[Tuple]):
    pass
def hBridge(compList: List[Tuple], conList: List[Tuple]):
    pass

def main():
    components = pd.import_components('./exported_data/data.json')
    mu = 20 #mü is the amount of average parts
    sigma = 5 #sigma is standart deviation

    partamount = int(np.random.normal(mu, sigma, 1))
    partcount = 0
    if partamount < 3:
        partamount = 3

    # randomly select if it should be a Component Line or SubComponent 
    offset = ()
    gridsize = 150
    #newBounding
    pos = ()
    compList: List[CirCmp] = []
    conList :List[Tuple[CirCmp, Pin, CirCmp, Pin]] = []
    unsatisfiedList = [] # sollte durch statisches array ersretzt werden
    #set the components
    rancols = random.randint(5, 10)
    ranrows = math.ceil(partamount / rancols)
    for i in range(rancols):
        for j in range(ranrows):
            componentSize = int(random.randint(64,128))
            pos = (
            i * gridsize + random.randint(-25, 25),
            j * gridsize + random.randint(-25, 25))

            random_type = random.sample(components.keys(), k=1)[0]
            cmp = random.sample(components[random_type], k=1)[0]

            cmp = cmp.load()  
            cmp.scale(componentSize / np.max(cmp.component_img.shape))
            cmp.rotate(random.randint(-5,5))

            newEntry = CirCmp(random_type, cmp, pos)
            compList.append(newEntry)


    for i in range(partamount):
        if len([*compList[i].cmp.pins.values()]) == 3:
            unsatisfiedList.append((i, 2))#third pin
        elif len([*compList[i].cmp.pins.values()]) == 1:
             unsatisfiedList.append((i, 0))#thirst pin
        else:
            print(i)
            conList.append((
               compList[i],
               [*compList[i].cmp.pins.values()][1],
               compList[i + 1],
               [*compList[i + 1].cmp.pins.values()][0]
            ))
    conList.append((
               compList[0],
               [*compList[0].cmp.pins.values()][1],
               compList[-1],
               [*compList[-1].cmp.pins.values()][0]
    ))

    print(len(unsatisfiedList))
    for i in range(len(unsatisfiedList)):
        print("Test")
        print(unsatisfiedList[i][0])
        conList.append((
              compList[unsatisfiedList[i - 1][0]],
              [*compList[unsatisfiedList[i - 1][0]].cmp.pins.values()][unsatisfiedList[i - 1][1]],
              compList[unsatisfiedList[i][0]],
              [*compList[unsatisfiedList[i][0]].cmp.pins.values()][unsatisfiedList[i][1]]
            ))

        
    print(len(conList)) 
    print(partamount)
    print(rancols)
    print(ranrows)
    cv2.imshow("reset",draw_routed_circuit(route(compList, conList)))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()