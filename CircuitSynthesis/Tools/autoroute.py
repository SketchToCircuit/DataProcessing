from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
import random
import math

from numpy.random.mtrand import normal

from .squigglylines import Lines
from .PinDetection.pindetection import Component, Pin, import_components

import cProfile

MIN_PIN_DIST = 50
MIN_LEN_LINE_CROSSING = 200

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
        img = np.full((2*self.radius, 2*self.radius), 255, dtype=np.uint8)
        cv2.circle(img, (self.radius, self.radius), self.radius, 0, cv2.FILLED)
        return img

@dataclass
class ConnLine:
    start: np.ndarray
    end: np.ndarray
    persistent: bool = False
    crossing: bool = False

    def to_img(self):
        thick = random.randint(2, 6)
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
    lines: List[ConnLine]
    knots: List[Knot] = None

    def offset_positions(self, offset):
        for knot in self.knots:
            knot.position -= offset
        for line in self.lines:
            line.start -= offset
            line.end -= offset
        for cmp in self.components:
            cmp.pos -= offset

    def convert_knot_cmp(self):
        self.knots = self.knots if self.knots else []
        for cmp in self.components:
            if cmp.type_id == 'knot':
                self.knots.append(Knot(cmp.pos - 10, 10))

        self.components[:] = [cmp for cmp in self.components if cmp.type_id != 'knot']

    def mark_lines_crossing(self):
        for line in self.lines:
            if np.sum(np.square(np.abs(line.end - line.start))) > MIN_LEN_LINE_CROSSING**2:
                line.crossing = True
    
    def remove_occupied(self):
        delete = []

        for knot in self.knots:
            for cmp in self.components:
                if _point_in_component(cmp, knot.position):
                    delete.append(True)
                    break
            else:
                delete.append(False)

        self.knots[:] = [k for d, k in zip(delete, self.knots) if not d]

        delete = []

        for line in self.lines:
            if line.persistent:
                delete.append(False)
                continue
            for cmp in self.components:
                if _point_in_component(cmp, line.start) or _point_in_component(cmp, line.end):
                    delete.append(True)
                    break
            else:
                delete.append(False)

        self.lines[:] = [l for d, l in zip(delete, self.lines) if not d]

from .render import *

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

def _point_in_component(cmp: CirCmp, p: np.ndarray):
    try:
        if p[0] > cmp.pos[0] and p[0] < cmp.pos[0] + cmp.cmp.component_img.shape[1]:
            if p[1] > cmp.pos[1] and p[1] < cmp.pos[1] + cmp.cmp.component_img.shape[0]:
                return True
    except Exception:
        return False
    return False
    
def _route_half_between_pos(pos_a: np.ndarray, pos_b: np.ndarray, no_go_dir: np.ndarray):
    dx, dy = pos_b - pos_a

    x_first: bool

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
        new_line = ConnLine(pos_a, pos_a + np.array([dx / 2.0, 0], dtype=float))
    else:
        new_line = ConnLine(pos_a, pos_a + np.array([0, dy / 2.0], dtype=float))

    return new_line

def _route_between_pos(components: List[CirCmp], pos_a: np.ndarray, pos_b: np.ndarray, no_go_dir_a: np.ndarray, no_go_dir_b: np.ndarray, conn_lines: List[ConnLine]):
    l_a = _route_half_between_pos(pos_a, pos_b, no_go_dir_a)
    l_b = _route_half_between_pos(pos_b, pos_a, no_go_dir_b)

    if np.argmin(np.abs((l_a.end - l_a.start))) == np.argmin(np.abs((l_b.end - l_b.start))):
        # both lines are either veritcal or horiontal -> one connection line needed
        conn_lines.append(ConnLine(l_a.end, l_b.end))
    else:
        # two lines are neede
        dx_a, _ = l_a.end - l_a.start

        if dx_a == 0:
            mid = [l_a.end[0], l_b.end[1]]
        else:
            mid = [l_b.end[0], l_a.end[1]]

        conn_lines.append(ConnLine(l_a.end, mid))
        conn_lines.append(ConnLine(l_b.end, mid))

    conn_lines.append(l_a)
    conn_lines.append(l_b)

def _route_single_components(components: List[CirCmp], connection: Tuple[CirCmp, Pin, CirCmp, Pin]):
    conn_lines: List[ConnLine] = []

    pos_a = connection[0].pos
    pos_b = connection[2].pos

    direct_line = False

    # if at least one pin is angled ~45°, make a direct connection -> allow bridge rectifier
    if connection[1]:
        k = abs(connection[1].direction[1]) / (abs(connection[1].direction[0]) + 0.0001)
        if k < 2.4 and k > 0.4:
            # angle (mapped to first quadrant) in [22.5°; 67.5°]
            direct_line = True
            pos_a = connection[1].position + connection[0].pos

    if connection[3]:
        k = abs(connection[3].direction[1]) / (abs(connection[3].direction[0]) + 0.0001)
        if k < 2.4 and k > 0.4:
            # angle (mapped to first quadrant) in [22.5°; 67.5°]
            direct_line = True
            pos_b = connection[3].position + connection[2].pos

    if direct_line:
        conn_lines = [ConnLine(pos_a, pos_b, persistent=True)]
        return conn_lines

    if connection[0].type_id != 'knot':
        # component one stud
        pos_a = connection[1].position + MIN_PIN_DIST * connection[1].direction + connection[0].pos
        conn_lines.append(ConnLine(connection[1].position + connection[0].pos, pos_a, persistent=True))

    if connection[2].type_id != 'knot':
        # component two stud
        pos_b = connection[3].position + MIN_PIN_DIST * connection[3].direction + connection[2].pos
        conn_lines.append(ConnLine(connection[3].position + connection[2].pos, pos_b, persistent=True))

    if pos_a is not None and pos_b is not None:
        _route_between_pos(components, pos_a, pos_b, -connection[1].direction if connection[1] is not None else None, -connection[3].direction if connection[3] is not None else None, conn_lines)

    return conn_lines

def _place_knots(components: List[CirCmp], connections: List[Tuple[CirCmp, Pin, CirCmp, Pin]]):
    knots = [cmp for cmp in components if cmp.type_id == 'knot' and cmp.pos is None]
    knot_conns = [conn for conn in connections if conn[0] in knots or conn[2] in knots]
    
    # place at (0, 0)
    for knot in knots:
        knot.pos = np.zeros(2, dtype=float)

    # go over all knots and connections and move them according to a "rubber-band" force
    for i in range(5):
        for knot in knots:
            force = np.zeros(2, dtype=float)
            for conn in knot_conns:
                if conn[0] is knot:
                    if conn[2].type_id == 'knot':
                        force += conn[2].pos - knot.pos
                    else:
                        force += conn[3].position + conn[2].pos - knot.pos
                elif conn[2] is knot:
                    if conn[0].type_id == 'knot':
                        force += conn[0].pos - knot.pos
                    else:
                        force += conn[1].position + conn[0].pos - knot.pos

            # reduce force per iteration to make placement more stable
            knot.pos += force / (3.0 + i)

def route(components: List[CirCmp], connections: List[Tuple[CirCmp, Pin, CirCmp, Pin]]) -> RoutedCircuit:
    conn_lines: List[ConnLine] = []
    knots: List[Knot] = []

    _place_knots(components, connections)

    for conn in connections:
        new_lines = _route_single_components(components, conn)
        conn_lines.extend(new_lines)

    circuit = RoutedCircuit(components, conn_lines)
    circuit.convert_knot_cmp()
    circuit.mark_lines_crossing()
    circuit.remove_occupied()
    return circuit

def main():
    from .export_tfrecords import export_circuits, export_label_map

    circuits = []
    for i in range(100):
        unload_cmp = import_components('./exported_data/data.json')

        #start_time = time.perf_counter()

        components = [CirCmp(t, r.load(), np.zeros(2)) for t in ['R', 'L', 'C', 'L2', 'C_P', 'F'] for i, r in enumerate(random.sample(unload_cmp[t], 3))]

        cmp_side_num = round(math.sqrt(len(components)))

        for i, cmp in enumerate(components):
            x, y = i // cmp_side_num, i % cmp_side_num
            cmp.pos = np.array([x, y]) * 300 + np.random.normal(0, 50, 2)
            cmp.cmp.scale((random.random() * 200 + 200) / np.max(cmp.cmp.component_img.shape))
            #cmp.cmp.rotate(45)

        connections = []
        #connections.append((components[0], components[0].cmp.pins['1'], components[1], components[1].cmp.pins['1']))
        #components.append(CirCmp('knot', None, None))
        #components.append(CirCmp('knot', None, None))
        #connections.append((components[1], components[1].cmp.pins['2'], components[-1], None))
        #connections.append((components[2], components[2].cmp.pins['1'], components[-1], None))
        #connections.append((components[-2], None, components[-1], None))
        #connections.append((components[-2], None, components[0], components[0].cmp.pins['2']))

        routed = route(components, connections)
        circuits.append(routed)
        # img = draw_routed_circuit(routed)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # print(f'{(time.perf_counter() - start_time) * 1000} ms')

    # cv2.imshow('', img)
    # cv2.waitKey()
    export_circuits(circuits, 'ObjectDetection/data/train.tfrecord', 'ObjectDetection/data/val.tfrecord')
    export_label_map('ObjectDetection/data/label_map.pbtxt')

if __name__ == '__main__':
    from .export_tfrecords import inspect_record
    main()
    inspect_record('ObjectDetection/data/train.tfrecord', 2)