from typing import List
import numpy as np
import cv2

from .autoroute import Knot, ConnLine, RoutedCircuit, CirCmp

def draw_without_lines(components: List[CirCmp], labels = False):
    if not len(components):
        return np.full((1, 1), 255, dtype=np.uint8)

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
    else:
        shape = max_pos - min_pos
        shape = shape[::-1]

    res_img = np.full(shape, 255, dtype=np.uint8)

    positions -= min_pos

    for img, pos, size in zip(images, positions, sizes):
        res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]] = np.minimum(img, res_img[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]])

    return res_img, min_pos

def draw_routed_circuit(circuit: RoutedCircuit, labels=False):
    images: List[np.ndarray] = []
    positions = []
    sizes = []

    cmp_img, cmp_pos = draw_without_lines(circuit.components, labels)
    images.append(cmp_img)
    positions.append(cmp_pos)
    sizes.append(cmp_img.shape[1::-1])

    for knot in circuit.knots:
        img = knot.to_img()
        images.append(img)
        positions.append(knot.position)
        sizes.append(img.shape[-1::-1])

    for line in circuit.lines:
        img = line.to_img()
        images.append(img)
        positions.append((line.start + line.end) / 2.0 - np.array(img.shape[-1::-1]) / 2.0)
        sizes.append(img.shape[-1::-1])

    positions = np.array(positions, dtype=int)
    sizes = np.array(sizes, dtype=int)

    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions + sizes, axis=0)

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