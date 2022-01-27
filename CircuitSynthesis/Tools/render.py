from typing import List, Set
import numpy as np
import cv2

from synthesis import RoutedCircuit

COMPONENT_BORDER = 30

def draw_routed_circuit(circuit: RoutedCircuit, labels=False):
    images: List[np.ndarray] = []
    mask_cmps: Set[int] = set()
    persistent: Set[int] = set()
    others: Set[int] = set()
    positions = []
    sizes = []

    for cmp in circuit.components:
        mask_cmps.add(len(images))
        images.append(cmp.cmp.component_img)
        positions.append(cmp.pos)
        sizes.append(cmp.cmp.component_img.shape[::-1])

        if labels:
            others.add(len(images))
            images.append(cmp.cmp.label_img)
            positions.append(cmp.pos + cmp.cmp.label_offset)
            sizes.append(cmp.cmp.label_img.shape[::-1])

    for knot in circuit.knots:
        others.add(len(images))
        img = knot.to_img()
        images.append(img)
        positions.append(knot.position)
        sizes.append(img.shape[::-1])

    for line in circuit.lines:
        if line.persistent:
            persistent.add(len(images))
        else:
            others.add(len(images))

        img = line.to_img()
        images.append(img)
        positions.append((line.start + line.end) / 2.0 - np.array(img.shape[::-1], dtype=int) / 2.0)
        sizes.append(img.shape[::-1])

    positions = np.array(positions, dtype=int)
    sizes = np.array(sizes, dtype=int)

    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions + sizes, axis=0)

    circuit.offset_positions(min_pos)

    result_shape = max_pos - min_pos
    result_shape = result_shape[::-1]

    res_img = np.full(result_shape, 255, dtype=np.uint8)
    mask_img = np.full(result_shape, 255, dtype=np.uint8)

    positions -= min_pos

    for i in mask_cmps:
        img = images[i]
        pos = positions[i]
        size = sizes[i]
        mask_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]] = np.where(img < 200, 0, mask_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]])

    mask_img = cv2.erode(mask_img, np.ones((COMPONENT_BORDER, COMPONENT_BORDER), dtype=np.uint8))

    for i in others:
        img = images[i]
        pos = positions[i]
        size = sizes[i]

        res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]] = np.minimum(img, res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]])

    res_img[mask_img == 0] = 255

    for i in persistent.union(mask_cmps):
        img = images[i]
        pos = positions[i]
        size = sizes[i]

        res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]] = np.minimum(img, res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]])        

    return res_img