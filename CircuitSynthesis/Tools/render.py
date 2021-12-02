from typing import List
import numpy as np
import cv2

from .autoroute import Knot, ConnLine, RoutedCircuit, CirCmp
from .augment import augment_cmp_img

def draw_routed_circuit(circuit: RoutedCircuit, labels=False):
    images: List[np.ndarray] = []
    positions = []
    sizes = []

    for cmp in circuit.components:
        images.append(augment_cmp_img(cmp.cmp.component_img))
        positions.append(cmp.pos)
        sizes.append(cmp.cmp.component_img.shape[1::-1])

        if labels:
            images.append(cmp.cmp.label_img)
            positions.append(cmp.pos + cmp.cmp.label_offset)
            sizes.append(cmp.cmp.label_img.shape[1::-1])

    for knot in circuit.knots:
        img = knot.to_img()
        images.append(img)
        positions.append(knot.position)
        sizes.append(img.shape[-1::-1])

    for line in circuit.lines:
        img = line.to_img()
        images.append(img)
        positions.append((line.start + line.end) / 2.0 - np.array(img.shape[-1::-1], dtype=int) / 2.0)
        sizes.append(img.shape[-1::-1])

    positions = np.array(positions, dtype=int)
    sizes = np.array(sizes, dtype=int)

    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions + sizes, axis=0)

    circuit.offset_positions(min_pos)

    if len(images[0].shape) == 3:
        shape = max_pos - min_pos
        shape = (shape[1], shape[0], images[0].shape[2])
    else:
        shape = max_pos - min_pos
        shape = shape[::-1]

    res_img = np.full(shape, 255, dtype=np.uint8)

    positions -= min_pos

    for img, pos, size in zip(images, positions, sizes):
        res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]] = np.minimum(img, res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]])

    return res_img