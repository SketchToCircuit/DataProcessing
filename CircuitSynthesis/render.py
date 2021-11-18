from typing import List
import numpy as np

import PinDetection.pindetection as pd

from synthesis import CirCmp

def draw_without_lines(components: List[CirCmp], labels = False):
    images: List[np.ndarray] = []
    positions = []
    sizes = []

    for cmp in components:
        if labels:
            images.append(cmp.cmp.component_img)
            positions.append(cmp.pos)
            sizes.append(cmp.cmp.component_img.shape[1::-1])

            images.append(cmp.cmp.label_img)
            positions.append(cmp.pos + cmp.cmp.label_offset)
            sizes.append(cmp.cmp.label_img.shape[1::-1])
        else:
            images.append(cmp.cmp.component_img)
            positions.append(cmp.pos)
            sizes.append(cmp.cmp.component_img.shape[1::-1])

    positions = np.array(positions)
    sizes = np.array(sizes)

    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions + sizes, axis=0)

    if len(images[0].shape) == 3:
        shape = max_pos - min_pos
        shape = (shape[1], shape[0], images[0].shape[2])
        print(shape)
    else:
        shape = max_pos - min_pos
        shape = shape[::-1]

    res_img = np.full(shape, 255, dtype=np.uint8)

    positions -= min_pos

    for img, pos, size in zip(images, positions, sizes):
        res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]] = np.minimum(img, res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]])
    
    return res_img