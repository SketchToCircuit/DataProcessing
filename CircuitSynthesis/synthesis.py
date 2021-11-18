from dataclasses import dataclass
from typing import Tuple, List
from PinDetection.pindetection import Component
import PinDetection.pindetection as pd
from squigglylines import * 

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

def bridgecircuit(componentsize):
    pass
def starcircuit(componentsize):
    pass
def deltacircuit(componentsize):
    pass

def main():
    components = pd.import_components('./exported_data/data.json')
    mu = 20 #mü is the amount of average parts
    sigma = 5 #sigma is standart deviation

    partamount = int(np.random.normal(mu, sigma, 1))
    partcount = 0
    if partamount < 3:
        partamount = 3

    componentSize = int(random.randint(64,128))

    # randomly select if it should be a Component Line or SubComponent 
    offset = ()
    gridsize = 128#50px
    #newBounding
    pos = ()
    compList = []
    conList = []
    #set the components
    for i in range(partamount):
        pos = (
        i * gridsize + random.randint(-50, 50),
        i * gridsize + random.randint(-50, 50)
        )
        if random.random() < 0.1:
            if random.random() < 0.33:
                newEntry = CirCmp(components.keys(), cmp, pos)
                compList.append(newEntry)
            elif random.random() < 0.66:
                pass
            else:
                pass
        else:
            random_type = random.sample(components.keys(), k=1)[0]
            cmp = random.sample(components[random_type], k=1)[0]

        newEntry = CirCmp(components.keys(), cmp, pos)
        compList.append(newEntry)


    
    
    

if __name__ == '__main__':
    main()