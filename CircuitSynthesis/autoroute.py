from typing import List
import numpy as np
import cv2
import random

import PinDetection.pindetection as pd

if __name__ == '__main__':
    from synthesis import CirCmp
    from render import *
else:
    from CircuitSynthesis.synthesis import CirCmp
    from render import *

def route(components: List[CirCmp]):
    raise NotImplementedError

def main():
    unload_cmp = pd.import_components('./exported_data/data.json')
    components = [CirCmp('NPN', r.load(), np.random.randint(600, size=(2))) for r in random.sample(unload_cmp['NPN'], 3)]
    
    for cmp in components:
        cmp.cmp.scale(200.0 / np.max(cmp.cmp.component_img.shape))

    cv2.imshow('', draw_without_lines(components))
    cv2.waitKey()
    #route(components)

if __name__ == '__main__':
    main()