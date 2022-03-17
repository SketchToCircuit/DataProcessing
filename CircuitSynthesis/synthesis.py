import random
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import dummy_sketch_generation as dsg
import Tools.PinDetection.pindetection as pd
from Tools.squigglylines import *

PART_COUNT_MU = 20 #mü is the amount of average parts
PART_COUNT_SIGMA = 5 #sigma is standart deviation

STUD_LENGTH = 70

MAX_GRIDSIZE_OFFSET = 30
GRIDSIZE = 150

MIN_LEN_LINE_CROSSING = 150

NUM_FILES = 20
CIRCUITS_PER_FILE = 1000
VALIDATION_NUM = 120
VAL_SRC_SPLIT = 0.1

DUMMY_SKETCH_GENERATION_PART = 0.35

DEBUG = False

@dataclass
class CirCmp:
    type_id: str
    cmp: pd.Component
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
    thick: ClassVar[int] = 2

    start: np.ndarray
    end: np.ndarray
    persistent: bool = False
    crossing: bool = False

    def to_img(self):
        change_thick = 0 if random.random() < 0.7 else 1
        ConnLine.thick = (ConnLine.thick - 1 + change_thick) % 2 + 1

        size = np.round(np.abs(self.start - self.end)).astype(int) + ConnLine.thick * 8 + 20
        img = np.full(size[::-1], 255, dtype=np.uint8)
        mid = (self.start + self.end) / 2.0
        a = self.start - mid + size / 2.0
        b = self.end - mid + size / 2.0

        if self.crossing and random.random() < 0.5:
            Lines.linecrossing(a[0], a[1], b[0], b[1], img, ConnLine.thick, 0)
        else:
            Lines.squigglyline(a[0], a[1], b[0], b[1], img, ConnLine.thick, 0)
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
                r = random.randint(2, 5)
                self.knots.append(Knot(cmp.pos - r, r))

        self.components[:] = [cmp for cmp in self.components if cmp.type_id != 'knot']

    def mark_lines_crossing(self):
        for line in self.lines:
            if np.sum(np.square(np.abs(line.end - line.start))) > MIN_LEN_LINE_CROSSING**2:
                line.crossing = True

from Tools.render import *

def _augment(component: pd.Component):

    if random.random() < 0.5:
        component.flip(vert=True)

    # some types have their own vertical version -> don't rotate 90°
    both_versions = ["A_H", "A_V", "U_AC_H", "U_AC_V", "V_H", "V_V", "M", "M_V"]
    # some types can be diagonal (45°)
    diagonal_allowed = ["C", "C_P", "D", "D_S", "D_Z", "L", "L2", "R"]

    angle = 0.0

    if component.type in both_versions:
        angle = random.choice([0.0, 180.0])
    elif component.type in diagonal_allowed:
        # only 10% are diagonal
        if random.random() < 0.1:
            angle = random.choice([45.0, -45.0, 135.0, -135.0])
        else:
            angle = random.choice([0.0, 90.0, -90.0, 180.0])
    else:
        angle = random.choice([0.0, 90.0, -90.0, 180.0])
    
    angle += np.random.normal(0, 5)

    component.rotate(angle)

def _finalize_components(components: List[CirCmp], width, height) -> RoutedCircuit:
    conn_lines: List[ConnLine] = []
    knots: List[CirCmp] = []

    for cmp in components:
        for pin in cmp.cmp.pins.values():
            pos = pin.position + cmp.pos
            direction = pin.direction

            if random.random() < 0.8:
                if random.random() < 0.5:
                    if random.random() < 0.5:
                        direction = pin.direction[::-1] * np.array([1, -1])
                    else:
                        direction = pin.direction[::-1] * np.array([-1, 1])
                # directly attached stud
                length = max(np.random.normal(STUD_LENGTH, 20), 5)
                pos += length * direction
                conn_lines.append(ConnLine(pin.position + cmp.pos, pos, persistent=True))
            
            # further branching
            if random.random() < 0.5:
                if random.random() < 0.7:
                    # simple corner
                    if random.random() < 0.5:
                        direction = direction[::-1] * np.array([1, -1])
                    else:
                        direction = direction[::-1] * np.array([1, -1])
                    length = max(np.random.normal(STUD_LENGTH * 1.5, 20), 5)
                    conn_lines.append(ConnLine(pos, pos + length * direction, persistent=False))
                else:
                    # junction
                    knots.append(CirCmp('knot', None, pos))
                    del_1 = max(np.random.normal(STUD_LENGTH * 2, 40), 5) * direction
                    del_2 = max(np.random.normal(STUD_LENGTH * 2, 40), 5) * direction[::-1] * np.array([1, -1])
                    del_3 = max(np.random.normal(STUD_LENGTH * 2, 40), 5) * direction[::-1] * np.array([-1, 1])

                    if random.random() < 0.2:
                        # 4 way junction
                        conn_lines.append(ConnLine(pos, pos + del_1, persistent=False))
                        conn_lines.append(ConnLine(pos, pos + del_2, persistent=False))
                        conn_lines.append(ConnLine(pos, pos + del_3, persistent=False))
                    else:
                        # drop one direction
                        a = random.random()
                        if a < 1/3:
                            conn_lines.append(ConnLine(pos, pos + del_1, persistent=False))
                            conn_lines.append(ConnLine(pos, pos + del_2, persistent=False))
                        elif a < 2/3:
                            conn_lines.append(ConnLine(pos, pos + del_1, persistent=False))
                            conn_lines.append(ConnLine(pos, pos + del_3, persistent=False))
                        else:
                            conn_lines.append(ConnLine(pos, pos + del_2, persistent=False))
                            conn_lines.append(ConnLine(pos, pos + del_3, persistent=False))

    components.extend(knots)
    circuit = RoutedCircuit(components, conn_lines)
    circuit.convert_knot_cmp()
    circuit.mark_lines_crossing()

    return circuit

def _create_circuit(components: Dict[str, pd.UnloadedComponent], validation=False):
    partamount = int(np.random.normal(PART_COUNT_MU, PART_COUNT_SIGMA, 1))
    if partamount < 3:
        partamount = 3

    pos = ()
    compList: List[CirCmp] = []
    # randomly define colums and rows
    rancols = random.randint(3, 7)
    ranrows = math.ceil(partamount / rancols)

    # If both versions are available -> make them half as likely to get choosen
    both_versions = ["A_H", "A_V", "U_AC_H", "U_AC_V", "V_H", "V_V", "M", "M_V"]
    weights = [0.5 if t in both_versions else 1.0 for t in components.keys()]

    for i in range(rancols):
        for j in range(ranrows):
            pos = (
            j * GRIDSIZE + random.randint(-MAX_GRIDSIZE_OFFSET, MAX_GRIDSIZE_OFFSET), #X
            i * GRIDSIZE + random.randint(-MAX_GRIDSIZE_OFFSET, MAX_GRIDSIZE_OFFSET)) #Y
            
            random_type = random.choices([*components.keys()], weights=weights, k=1)[0]
            num_val = int(len(components[random_type]) * VAL_SRC_SPLIT)

            if validation:
                rand_idx = random.randint(0, num_val - 1)
            else:
                rand_idx = random.randint(num_val - 1, len(components[random_type]) - 1)
            
            cmp = components[random_type][rand_idx]
            
            #Loaded components are enabled to Edit
            cmp = cmp.load()

            #make some components bigger[by now only the OPV]
            bigger = ["OPV"]
            componentSize = int(random.randint(64, 128))
            if cmp.type in bigger:
                cmp.scale((componentSize + 40) / np.max(cmp.component_img.shape))
            else:
                cmp.scale(componentSize / np.max(cmp.component_img.shape))

            _augment(cmp)

            newEntry = CirCmp(random_type, cmp, pos)
            compList.append(newEntry)

    return _finalize_components(compList, ranrows * GRIDSIZE, rancols * GRIDSIZE)

if __name__ == '__main__':
    from Tools.export_tfrecords import export_circuits, export_label_map, _parse_fine_to_coarse

    export_label_map('./DataProcessing/ObjectDetection/data/label_map.pbtxt', './DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')

    components = pd.import_components('./DataProcessing/pindetection_data/data.json')

    label_convert = _parse_fine_to_coarse('./DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')

    if DEBUG:
        for i in range(5):
            circ = _create_circuit(components)
            cv2.imshow('', draw_routed_circuit(circ, labels=True))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        exit()

    num_dummy_circuit_data = (len(os.listdir('./DataProcessing/CircuitSynthesis/DummySketchData/')) - 1) // 2

    val_cirucits: List[RoutedCircuit] = []
    val_dummy_circuits = []

    for i in range(VALIDATION_NUM):
        if random.random() < DUMMY_SKETCH_GENERATION_PART:
            x = random.randint(0, num_dummy_circuit_data - 1)
            val_dummy_circuits.extend(dsg.get_examples(f'./DataProcessing/CircuitSynthesis/DummySketchData/{x}_dummy.jpg', f'./DataProcessing/CircuitSynthesis/DummySketchData/{x}_mask.jpg', components, label_convert))
        else:
            val_cirucits.append(_create_circuit(components, validation=True))

        if i % 100 == 0:
            print(f'val:{i}')
    
    export_circuits(val_cirucits, f'./DataProcessing/ObjectDetection/data/val.tfrecord', val_dummy_circuits, './DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')

    for f in range(NUM_FILES):
        cirucits: List[RoutedCircuit] = []
        dummy_circuits = []
        
        for i in range(CIRCUITS_PER_FILE):
            if random.random() < DUMMY_SKETCH_GENERATION_PART:
                x = random.randint(0, num_dummy_circuit_data - 1)
                dummy_circuits.extend(dsg.get_examples(f'./DataProcessing/CircuitSynthesis/DummySketchData/{x}_dummy.jpg', f'./DataProcessing/CircuitSynthesis/DummySketchData/{x}_mask.jpg', components, label_convert))
            else:
                cirucits.append(_create_circuit(components))

            if i % 200 == 0:
                print(f'{f}:{i}')

        export_circuits(cirucits, f'./DataProcessing/ObjectDetection/data/train-{f}.tfrecord', dummy_circuits, './DataProcessing/ObjectDetection/fine_to_coarse_labels.txt')
