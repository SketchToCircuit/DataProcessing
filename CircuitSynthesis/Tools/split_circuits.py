from typing import List, Tuple

from CircuitSynthesis.Tools.squigglylines import Lines
from .autoroute import CirCmp, RoutedCircuit

import numpy as np
import cv2
import random
import math

SUB_SIZE = 800
MIN_OVERLAP = 50

def split_circuit(bboxs: List[Tuple[float, float, float, float]], img: np.ndarray):
    '''bboxs: list of (xmin, ymin, xmax, ymax)'''

    size = np.array(img.shape)[::-1]

    if np.all(size < SUB_SIZE):
        print('No splitting')
        return [(bboxs, img)]

    num_subs_x = 2
    overlap_x = 0
    while True:
        overlap_x = (num_subs_x * SUB_SIZE - size[0]) / (num_subs_x - 1)

        if overlap_x > MIN_OVERLAP:
            break

        num_subs_x += 1
    
    num_subs_y = 2
    overlap_y = 0
    while True:
        overlap_y = (num_subs_y * SUB_SIZE - size[1]) / (num_subs_y - 1)

        if overlap_y > MIN_OVERLAP:
            break
        
        num_subs_y += 1
    
    result: List[Tuple[List[Tuple[float,float, float, float]] , np.ndarray]] = []

    for i_x in range(num_subs_x):
        for i_y in range(num_subs_y):
            start = np.maximum(np.array([i_x, i_y]) * (SUB_SIZE - np.array([overlap_x, overlap_y])), 0).astype(int)
            end = np.minimum(start + SUB_SIZE, size).astype(int)

            new_img = img[start[1]:end[1], start[0]:end[0]]
            new_bboxs = []

            for bbox in bboxs:
                new_bboxs.append((bbox[0] - start[0], bbox[1] - start[1], bbox[2] - start[0], bbox[3] - start[1]))
            
            result.append((new_bboxs, new_img))

    return result