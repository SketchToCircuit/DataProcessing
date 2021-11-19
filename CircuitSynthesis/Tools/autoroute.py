from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
import random

from .squigglylines import Lines
from .PinDetection.pindetection import Component, Pin, import_components

@dataclass
class CirCmp:
    type_id: str
    cmp: Component
    pos: np.ndarray

@dataclass
class Knot:
    position: np.ndarray
    radius: int

    def to_img(self):
        img = np.full((2*self.radius, 2*self.radius), 255)
        cv2.circle(img, (self.radius, self.radius), self.radius, 0, cv2.FILLED)
        return img

@dataclass
class ConnLine:
    start: np.ndarray
    end: np.ndarray
    crossing: bool = False

    def to_img(self):
        thick = 3
        size = np.abs((self.start - self.end)) + thick * 8 + 20
        img = np.full(size[::-1], 255, dtype=np.uint8)
        mid = (self.start + self.end) / 2.0
        a = self.start - mid + size / 2.0
        b = self.end - mid + size / 2.0

        if self.crossing and random.random() < 0.5:
            Lines.linecrossing(a[0], a[1], b[0], b[1], img, thick, 0)
        else:
            Lines.squigglyline(a[0], a[1], b[0], b[1], img, thick, 0)
        return img

@dataclass
class RoutedCircuit:
    components: List[CirCmp]
    knots: List[Knot]
    lines: List[ConnLine]

    def remove_knots(self):
        self.components = [cmp for cmp in self.components if cmp.type_id != 'knot']

from .render import *

MIN_PIN_DIST = 50

def route_single(components: List[CirCmp], connection: Tuple[CirCmp, Pin, CirCmp, Pin]):
    conn_lines: List[ConnLine] = []
    knots: List[Knot] = []

    if connection[0].type_id != 'knot' and connection[2].type_id != 'knot':
        # component one stud
        ext_a = connection[1].position + MIN_PIN_DIST * connection[1].direction + connection[0].pos
        conn_lines.append(ConnLine(connection[1].position + connection[0].pos, ext_a))

        # component two stud
        ext_b = connection[3].position + MIN_PIN_DIST * connection[3].direction + connection[2].pos
        conn_lines.append(ConnLine(connection[3].position + connection[2].pos, ext_b))

        conn_lines.append(ConnLine(ext_a, ext_b))

    return (knots, conn_lines)

def route(components: List[CirCmp], connections: List[Tuple]) -> RoutedCircuit:
    conn_lines: List[ConnLine] = []
    knots: List[Knot] = []

    for conn in connections:
        new_knots, new_lines = route_single(components, conn)
        conn_lines.extend(new_lines)
        knots.extend(new_knots)
    
    circuit = RoutedCircuit(components, knots, conn_lines)
    circuit.remove_knots()
    return circuit

def main():
    unload_cmp = import_components('./exported_data/data.json')
    components = [CirCmp('R', r.load(), np.random.randint(600, size=(2))) for r in random.sample(unload_cmp['R'], 2)]

    for cmp in components:
        cmp.cmp.scale(400.0 / np.max(cmp.cmp.component_img.shape))

    routed = route(components, [(components[0], components[0].cmp.pins['1'], components[1], components[1].cmp.pins['1'])])

    img = draw_routed_circuit(routed)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.imshow('', img)
    cv2.waitKey()

if __name__ == '__main__':
    main()