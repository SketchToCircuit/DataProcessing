import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

# information of one pin
@dataclass
class Pin:
    name: str
    '''Name of the pin (eg. "gate", "1", "out", ...)'''
    position: np.ndarray
    '''Position of the pin as 1D numpy array with size 2: [x, y]'''
    direction: np.ndarray
    '''Direction of the pin as 1D numpy array with size 2: [x_dir, y_dir]'''

    def to_dict(self):
        '''Generate a dict object to export as JSON.'''
        return {'name': self.name, 'position': self.position.tolist(), 'direction': self.direction.tolist()}
    
    @staticmethod
    def from_dict(map):
        '''Generate a class instance from a dict object.'''
        try:
            return Pin(map['name'], np.array(map['position']), np.array(map['direction']))
        except Exception:
            return Pin('0', np.zeros(2), np.zeros(2))

# interface for pin detector
class IPinDetection(ABC):
    @property
    @classmethod
    @abstractmethod
    def _COMP_TYPES(cls): raise NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def NAME(cls): raise NotImplementedError

    @classmethod
    def match(cls, component_type: str) -> bool :
        return component_type in cls._COMP_TYPES
    
    @staticmethod
    @abstractmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img: np.ndarray) -> Dict[str, Pin]:
        raise NotImplementedError

class TwoPinsHor(IPinDetection):
    _COMP_TYPES = ['R', 'C', 'L', 'L2', 'R_H', 'C_H', 'L_H',
    'LED', 'D', 'S1', 'S2', 'BTN1', 'BTN2', 'V_H', 'A_H', 'U_AC_H',
    'LMP', 'M', 'F', 'D_Z', 'D_S', 'C_P']
    NAME = 'TwoPinsHorizontale'

    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # calculate dx and dy
        deltas = np.diff(lines, axis=-2).squeeze(axis=-2)
        # create mask to remove elements where dy > dx
        mask = (np.diff(np.abs(deltas)) < 0).squeeze(axis=-1)
        lines = lines[mask, :, :]

        # get points with minimum and maximum of: x - abs(y - cy) or x + abs(y - cy) to take lines near to vertical center
        values = lines[:, :, 0] + np.abs(lines[:, :, 1] - centroid[1])
        left_point = lines[np.unravel_index(np.argmin(values), values.shape[0:2])]
        values = lines[:, :, 0] - np.abs(lines[:, :, 1] - centroid[1])
        right_point = lines[np.unravel_index(np.argmax(values), values.shape[0:2])]

        left_point[0] = left_point[0] + img_size[1] / 30
        right_point[0] = right_point[0] - img_size[1] / 30
        
        return {'1': Pin('1', left_point, np.array([-1, 0])), '2': Pin('2', right_point, np.array([1, 0]))}

class TwoPinsVert(IPinDetection):
    _COMP_TYPES = ['R_V', 'C_V', 'L_V',
    'V_V', 'A_V', 'U1', 'U2', 'I1', 'I2', 'U3', 'BAT', 'U_AC_V', 'M_V']
    NAME = 'TwoPinsVertical'
    
    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # rotate and then use other class
        rot_lines = lines[:, :, ::-1] * [1, -1]
        rot_centroid = centroid[::-1] * [1, -1]

        pins = TwoPinsHor.get_pins(rot_lines, rot_centroid, img_size[::-1])

        for pin in pins.values():
            pin.position = (pin.position * [1, -1])[::-1]
            pin.direction = (pin.direction * [1, -1])[::-1]

        return pins

class OnePinLeft(IPinDetection):
    _COMP_TYPES = ['PIN']
    NAME = 'OnePinLeft'
    
    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # filter vertical lines with dx and dy
        deltas = np.diff(lines, axis=-2).squeeze(axis=-2)
        mask = (np.diff(np.abs(deltas)) < 0).squeeze(axis=-1)
        lines = lines[mask, :, :]

        # get point with minimum of: x + abs(y - cy) to take lines near to vertical center
        values = lines[:, :, 0] + np.abs(lines[:, :, 1] - centroid[1])
        left_point = lines[np.unravel_index(np.argmin(values), values.shape[0:2])]

        left_point[0] = left_point[0] + img_size[1] / 30
        
        return {'1': Pin('1', left_point, np.array([-1, 0]))}

class OnePinTop(IPinDetection):
    _COMP_TYPES = ['GND', 'GND_F', 'GND_C']
    NAME = 'OnePinTop'
    
    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # rotate and use other class
        rot_lines = lines[:, :, ::-1] * [1, -1]
        rot_centroid = centroid[::-1] * [1, -1]

        pin = OnePinLeft.get_pins(rot_lines, rot_centroid, img_size[::-1])['1']

        pin.position = (pin.position * [1, -1])[::-1]
        pin.direction = np.array([0, -1])

        return {'1': pin}

class TwoPinsLeft(IPinDetection):
    _COMP_TYPES = ['SPK', 'MIC']
    NAME = 'TwoPinsLeft'
    
    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # filter vertical lines
        deltas = np.diff(lines, axis=-2).squeeze(axis=-2)
        mask = (np.diff(np.abs(deltas)) < 0).squeeze(axis=-1)
        lines = lines[mask, :, :]

        # get first point (farthest to the left) with x + abs(y -cy) as metric
        values = lines[:, :, 0] + np.abs(lines[:, :, 1] - centroid[1])
        left_1_idx = np.unravel_index(np.argmin(values), values.shape[0:2])
        left_1 = lines[left_1_idx]

        # calculate verrtical bounding area around first used line with margin
        min_y, max_y = np.sort(lines[left_1_idx[0]][:, 1])
        min_y -= img_size[1] / 50
        max_y += img_size[1] / 50

        # remove already used line
        lines = np.delete(lines, left_1_idx[0], axis=0)

        # create mask for all lines where both endpoints are outside of the bounding rect
        mask = np.logical_or(lines[:, :, 1] < min_y , lines[:, :, 1] > max_y)
        mask = np.logical_and(mask[:, 0], mask[:, 1])
        lines = lines[mask, :, :]

        # get second point
        values = lines[:, :, 0] + np.abs(lines[:, :, 1] - centroid[1])
        left_2_idx = np.unravel_index(np.argmin(values), values.shape[0:2])
        left_2 = lines[left_2_idx]

        left_1[0] = left_1[0] + img_size[1] / 30
        left_2[0] = left_2[0] + img_size[1] / 30

        return {'1': Pin('1', left_1, np.array([-1, 0])), '2': Pin('2', left_2, np.array([-1, 0]))}

class TwoLOneRHor(IPinDetection):
    _COMP_TYPES = ['OPV']
    NAME = 'TwoLeftOneRight'
    
    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # flipped lines: image size - coordinate (correct y coordinate afterwards)
        lines_flip = (img_size[::-1] * [1, 0] - lines[:, :]) * [1, -1]
        # flip centroid
        centroid_flip = [img_size[1] - centroid[0], centroid[1]]

        inputs = [*TwoPinsLeft.get_pins(lines, centroid, img_size).values()]
        out = OnePinLeft.get_pins(lines_flip, centroid_flip, img_size)['1']

        # flip right pin back and rename
        out.position[0] = img_size[1] - out.position[0]
        out.direction = np.array([1, 0])
        out.name = 'out'

        # first input is higher -> negative
        if inputs[0].position[1] < inputs[1].position[1]:
            inputs[0].name = '-'
            inputs[1].name = '+'

            return {'+': inputs[1], '-': inputs[0], 'out': out}
        else:
            inputs[0].name = '+'
            inputs[1].name = '-'
            return {'+': inputs[0], '-': inputs[1], 'out': out}

class TwoROneLHor(IPinDetection):
    _COMP_TYPES = ['S3']
    NAME = 'TwoRightOneLeft'
    
    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # flip img and lines
        lines_flip = lines
        lines_flip = (img_size[::-1] * [1, 0] - lines[:, :]) * [1, -1]
        centroid_flip = [img_size[1] - centroid[0], centroid[1]]

        pins = TwoLOneRHor.get_pins(lines_flip, centroid_flip, img_size)

        # flip back
        for pin in pins.values():
            pin.position[0] = img_size[1] - pin.position[0]
            pin.direction = pin.direction * [-1, 0]
        
        return pins

class Potentiometer(IPinDetection):
    _COMP_TYPES = ['POT']
    NAME = 'Potentiometer'

    @staticmethod
    def _get_pin_diagonal_bottom(lines: np.ndarray, centroid: np.ndarray, angles: np.ndarray, img_size: np.ndarray):
        # create mask for elements where angle in [10°; 90°]
        mask = np.abs(angles * 180 / math.pi - 50) < 41
        lines = lines[mask, :, :]

        # get point with maximum of: 2 * y - abs(x - cx) to take lines near to horizontale center
        values = 2 * lines[:, :, 1] - np.abs(lines[:, :, 0] - centroid[0])
        bottom_point = lines[np.unravel_index(np.argmax(values), values.shape[0:2])]

        bottom_point[1] = bottom_point[1] - img_size[0] / 30
        bottom_point[0] = bottom_point[0] + img_size[0] / 30
        
        return Pin('3', bottom_point, np.array([0, 1]))

    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        # get left and right pins
        pins = TwoPinsHor.get_pins(lines, centroid, img_size)

        # find diagonal lines
        deltas = np.abs(np.diff(lines, axis=-2).squeeze(axis=-2))

        # create mask for elements where atan2(dy, dx) in [15°; 80°]
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        mask = np.abs(angles * 180 / math.pi - 47.5) < 33
        diagonal_lines = lines[mask, :, :]

        # dx * dy > 0 -> correct quadrant of diagonal
        deltas = np.diff(diagonal_lines, axis=-2).squeeze(axis=-2)
        mask = (deltas[:, 0] * deltas[:, 1]) < 0
        diagonal_lines = diagonal_lines[mask, :, :]

        # find lines where one point is above and one is below the center and one is left and one right
        below_c = (diagonal_lines[:, :, 1] - centroid[1]) * 1.5 > 0
        mask = np.logical_xor(below_c[:, 0], below_c[:, 1])
        diagonal_lines = diagonal_lines[mask, :, :]

        left_c = diagonal_lines[:, :, 0] < centroid[0]
        mask = np.logical_xor(left_c[:, 0], left_c[:, 1])
        diagonal_lines = diagonal_lines[mask, :, :]

        # mask for lines near horizontale center
        mask = np.abs((diagonal_lines[:, 0, 0] + diagonal_lines[:, 1, 0]) / 2 - centroid[0]) < img_size[1] / 5

        if np.count_nonzero(mask) > 0:
            # diagonal line in the middle -> pin on bottom
            # shift centroid to the left and flip vertically
            centroid[0] = centroid[0] - img_size[1] / 15

            pins['3'] = Potentiometer._get_pin_diagonal_bottom(lines, centroid, angles, img_size)
        else:
            # pin on top
            pins['3'] = OnePinTop.get_pins(lines, centroid, img_size)['1']
            pins['3'].name = '3'

        return pins

class TransistorFET(IPinDetection):
    _COMP_TYPES = ['JFET_N', 'JFET_P', 'MFET_N_D', 'MFET_N_E',
    'MFET_P_D', 'MFET_P_E']
    NAME = 'TransistorFET'

    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        pin_points = _Transistor.get_pin_points(lines, centroid, img_size)
        return {'gate': Pin('gate', pin_points['l'], np.array([-1, 0])), 'drain': Pin('drain', pin_points['tr'], np.array([0, -1])), 'source': Pin('source', pin_points['br'], np.array([0, 1]))}

class TransistorBipolar(IPinDetection):
    _COMP_TYPES = ['NPN', 'PNP']
    NAME = 'TransistorBipolar'

    @staticmethod
    def get_pins(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        pin_points = _Transistor.get_pin_points(lines, centroid, img_size)
        return {'base': Pin('base', pin_points['l'], np.array([-1, 0])), 'collector': Pin('collector', pin_points['tr'], np.array([0, -1])), 'emitter': Pin('emitter', pin_points['br'], np.array([0, 1]))}

class _Transistor:
    @staticmethod
    def get_pin_points(lines: np.ndarray, centroid: np.ndarray, img_size: np.ndarray):
        left = OnePinLeft.get_pins(lines, centroid, img_size)['1']

        # filter horizontale lines
        deltas = np.abs(np.diff(lines, axis=-2).squeeze(axis=-2))
        # create mask for elements where atan2(dy, dx) in [20°; 90°]
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        mask = np.abs(angles * 180 / math.pi - 55) < 36
        lines = lines[mask, :, :]

        # move centroid to the right to center for the right pins
        centroid[0] += img_size[1] / 4

        # get point with minimum of: y + abs(x - cx) to take lines near to horizontale center
        values = lines[:, :, 1] + np.abs(lines[:, :, 0] - centroid[0])
        top_point = lines[np.unravel_index(np.argmin(values), values.shape[0:2])]
        # get point with maximum of: y - abs(x - cx) to take lines near to horizontale center
        values = lines[:, :, 1] - np.abs(lines[:, :, 0] - centroid[0])
        bottom_point = lines[np.unravel_index(np.argmax(values), values.shape[0:2])]

        top_point[1] = top_point[1] + img_size[0] / 30
        bottom_point[1] = bottom_point[1] - img_size[0] / 30

        return {'l': left.position, 'tr': top_point, 'br': bottom_point}

ALL_DETECTORS: List[IPinDetection] = [TwoPinsHor, TwoPinsVert, OnePinLeft, OnePinTop, TwoPinsLeft, TwoLOneRHor, TwoROneLHor, Potentiometer, TransistorFET, TransistorBipolar]
