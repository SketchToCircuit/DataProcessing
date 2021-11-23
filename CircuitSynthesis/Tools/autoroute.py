from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
import random

from numpy.random.mtrand import normal

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
        size = np.round(np.abs(self.start - self.end)).astype(int) + thick * 8 + 20
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
        temp = [cmp for cmp in self.components if cmp.type_id != 'knot']
        self.components = temp

from .render import *

MIN_PIN_DIST = 50

def _check_dir(dir: list, no_go_dir: list):
    if no_go_dir is None:
        return True

    cross = dir[0]*no_go_dir[0] + dir[1]*no_go_dir[1]

    return abs(cross - 1) > 0.3

def _seg_seg_coll(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    t_num = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    u_num = (x1-x3)*(y3-y4) - (y1-y2)*(x3-x4)

    if abs(denom) < 0.001:
        return None
    else:
        t = t_num / denom
        if  t > 0 and t < 1:
            u = u_num / denom
            if u > 0 and u < 1:
                return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    return None

def _check_collision(cmp: CirCmp, pos_a: np.ndarray, pos_b: np.ndarray):
    p1 = _seg_seg_coll(pos_a[0], pos_a[1], pos_b[0], pos_b[1], cmp.pos[0], cmp.pos[1], cmp.pos[0] + cmp.cmp.component_img.shape[1], cmp.pos[1])
    p2 = _seg_seg_coll(pos_a[0], pos_a[1], pos_b[0], pos_b[1], cmp.pos[0], cmp.pos[1], cmp.pos[0], cmp.pos[1] + cmp.cmp.component_img.shape[0])
    p3 = _seg_seg_coll(pos_a[0], pos_a[1], pos_b[0], pos_b[1], cmp.pos[0] + cmp.cmp.component_img.shape[1], cmp.pos[1], cmp.pos[0] + cmp.cmp.component_img.shape[1], cmp.pos[1] + cmp.cmp.component_img.shape[0])
    p4 = _seg_seg_coll(pos_a[0], pos_a[1], pos_b[0], pos_b[1], cmp.pos[0], cmp.pos[1] + cmp.cmp.component_img.shape[0], cmp.pos[0] + cmp.cmp.component_img.shape[1], cmp.pos[1] + cmp.cmp.component_img.shape[0])
    
    if p1 is not None or p2 is not None or p3 is not None or p4 is not None:
        result = [p for p in [p1, p2, p3, p4] if p is not None]
        return result
    return None
        
def _route_between_pos(components: List[CirCmp], pos_a: np.ndarray, pos_b: np.ndarray, no_go_dir: list, conn_lines: List[ConnLine]):
    dx, dy = pos_b - pos_a
    if dx**2 + dy**2 < 10:
        return

    x_first = None

    if abs(dx) > abs(dy):
        if _check_dir([np.sign(dx), 0], no_go_dir):
            x_first = True
        else:
            x_first = False
    else:
        if _check_dir([0, np.sign(dy)], no_go_dir):
            x_first = False
        else:
            x_first = True

    if x_first:
        new_line = ConnLine(pos_a, pos_a + np.array([dx, 0]))
        conn_lines.append(new_line)
        _route_between_pos(components, new_line.end, pos_b, None, conn_lines)
    else:
        new_line = ConnLine(pos_a, pos_a + np.array([0, dy]))
        conn_lines.append(new_line)
        _route_between_pos(components, new_line.end, pos_b, None, conn_lines)

def _route_single_components(components: List[CirCmp], connection: Tuple[CirCmp, Pin, CirCmp, Pin]):
    conn_lines: List[ConnLine] = []

    no_go_dir = connection[1].direction if connection[1] is not None else None
    pos_a = connection[0].pos
    pos_b = connection[2].pos

    if connection[0].type_id != 'knot':
        # component one stud
        pos_a = connection[1].position + MIN_PIN_DIST * connection[1].direction + connection[0].pos
        conn_lines.append(ConnLine(connection[1].position + connection[0].pos, pos_a))

    if connection[2].type_id != 'knot':
        # component two stud
        pos_b = connection[3].position + MIN_PIN_DIST * connection[3].direction + connection[2].pos
        conn_lines.append(ConnLine(connection[3].position + connection[2].pos, pos_b))

    if pos_a is not None and pos_b is not None:
        _route_between_pos(components, pos_a, pos_b, None, conn_lines)

    return conn_lines

def route(components: List[CirCmp], connections: List[Tuple[CirCmp, Pin, CirCmp, Pin]]) -> RoutedCircuit:
    conn_lines: List[ConnLine] = []
    knots: List[Knot] = []

    for conn in connections:
        new_lines = _route_single_components(components, conn)
        conn_lines.extend(new_lines)

        if conn[0].type_id == 'knot' and conn[0].pos is not None:
            knots.append(Knot(conn[0].pos - 10, 10))

        if conn[2].type_id == 'knot' and conn[2].pos is not None:
            knots.append(Knot(conn[2].pos - 10, 10))

    circuit = RoutedCircuit(components, knots, conn_lines)
    circuit.remove_knots()
    return circuit

def main():
    unload_cmp = import_components('./exported_data/data.json')
    components = [CirCmp('R', r.load(), np.random.randint(600, size=(2))) for r in random.sample(unload_cmp['R'], 3)]

    for cmp in components:
        cmp.cmp.scale(400.0 / np.max(cmp.cmp.component_img.shape))

    connections = []
    connections.append((components[0], components[0].cmp.pins['1'], components[1], components[1].cmp.pins['1']))
    components.append(CirCmp('knot', None, None))
    components.append(CirCmp('knot', None, None))
    connections.append((components[1], components[1].cmp.pins['2'], components[-1], None))
    connections.append((components[2], components[2].cmp.pins['1'], components[-1], None))
    connections.append((components[-2], None, components[-1], None))
    connections.append((components[-2], None, components[0], components[0].cmp.pins['2']))
    routed = route(components, connections)

    img = draw_routed_circuit(routed)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.imshow('', img)
    cv2.waitKey()

if __name__ == '__main__':
    main()