from dataclasses import dataclass
from typing import Tuple, List
from PinDetection.pindetection import Component
import PinDetection.pindetection as pd
from squigglylines import * 
from render import *

import random
import numpy as np

class CirCmp:
    type_id: str
    cmp: Component
    pos: np.ndarray

    def __init__(self, type_id, cmp, pos):
        self.type_id = type_id
        self.cmp = cmp
        self.pos = pos

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

    compList.append(CirCmp("knot", None, (
        ,
    ))
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
    gridsize = 128#50px
    #newBounding
    pos = ()
    compList = []
    conList = []
    #set the components
    for i in range(partamount):
        if random.random() < 0.05:
            if random.random() < 0.33:
                cmp = bridgecircuit()
            elif random.random() < 0.66:
                pass
            else:
                pass
        else:
            componentSize = int(random.randint(64,128))
            pos = (
            i * gridsize + random.randint(-50, 50),
            i * gridsize + random.randint(-50, 50))

            random_type = random.sample(components.keys(), k=1)[0]
            cmp = random.sample(components[random_type], k=1)[0]

            cmp = cmp.load()  
            cmp.scale(componentSize / np.max(cmp.component_img.shape))
            cmp.rotate(random.randint(-5,5))

        newEntry = CirCmp(components.type(), cmp, pos)
        compList.append(newEntry)


    
    
    

if __name__ == '__main__':
    main()