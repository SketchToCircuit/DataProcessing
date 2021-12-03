from typing import List, Tuple

from .squigglylines import Lines
from .autoroute import CirCmp, RoutedCircuit

import numpy as np
import cv2
import random
import math

SUB_SIZE = 640
MIN_OVERLAP = 50
MIN_COMPONENT_AREA_REL = 0.5

def split_circuit(bboxs: List[Tuple[float, float, float, float]], img: np.ndarray):
    '''
    bboxs: list of (xmin, ymin, xmax, ymax)
    returned bboxs are in [0; 1]
    '''

    size = np.array(img.shape)[::-1]

    if np.all(size < SUB_SIZE):
        img_h, img_w = img.shape
        return [([(bbox[0] / img_w,
                    bbox[1] / img_h,
                    bbox[2] / img_w,
                    bbox[3] / img_h) for bbox in bboxs], list(range(len(bboxs))), img)]

    num_subs_x = 2
    overlap_x = 0
    while True:
        overlap_x = (num_subs_x * SUB_SIZE - size[0]) / (num_subs_x - 1) / 2

        if overlap_x > MIN_OVERLAP:
            break

        num_subs_x += 1
    
    num_subs_y = 2
    overlap_y = 0
    while True:
        overlap_y = (num_subs_y * SUB_SIZE - size[1]) / (num_subs_y - 1) / 2

        if overlap_y > MIN_OVERLAP:
            break
        
        num_subs_y += 1
    
    result: List[Tuple[List[Tuple[float,float, float, float]], List[int], np.ndarray]] = []

    for i_x in range(num_subs_x):
        for i_y in range(num_subs_y):
            start = np.maximum(np.array([i_x, i_y]) * (SUB_SIZE - np.array([overlap_x, overlap_y])), 0).astype(int)
            end = np.minimum(start + SUB_SIZE, size).astype(int)

            new_img = img[start[1]:end[1], start[0]:end[0]]
            img_h, img_w = new_img.shape
            new_bboxs = []
            indices = []

            for i, bbox in enumerate(bboxs):
                new_box = (np.clip((bbox[0] - start[0]) / img_w, 0, 1),
                    np.clip((bbox[1] - start[1]) / img_h, 0, 1),
                    np.clip((bbox[2] - start[0]) / img_w, 0, 1),
                    np.clip((bbox[3] - start[1]) / img_h, 0, 1))

                inter_area = (np.clip(new_box[2], 0, 1) - np.clip(new_box[0], 0, 1)) * (np.clip(new_box[3], 0, 1) - np.clip(new_box[1], 0, 1))
                tot_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / img_h / img_w

                if inter_area > tot_area * MIN_COMPONENT_AREA_REL:
                    new_bboxs.append(new_box)
                    indices.append(i)
            
            result.append((new_bboxs, indices, new_img))

    return result