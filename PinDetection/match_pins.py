import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

# information of one pin
@dataclass
class Pin:
    name: str
    position: np.ndarray
    direction: np.ndarray

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
    def get_pins(lines: np.ndarray, img: np.ndarray) -> List[Pin]:
        raise NotImplementedError

class TwoPinsHor(IPinDetection):
    _COMP_TYPES = ['R', 'C', 'L', 'R_H', 'C_H', 'L_H',
    'LED', 'D', 'S1', 'S2', 'BTN1', 'BTN2', 'V_H', 'A_H', 'U_AC_H',
    'LMP', 'M', 'F', 'D_Z', 'D_S', 'C_P']

    NAME = 'TwoPinsHorizontale'

    @staticmethod
    def get_pins(lines: np.ndarray, img: np.ndarray):
        # calculate dx and dy
        deltas = np.diff(lines, axis=-2).squeeze()
        # create mask to remove elements where dy > dx
        mask = (np.diff(np.abs(deltas)) < 0).squeeze()
        lines = lines[mask, :, :]

        # get points with minimum and maximum x-value
        left_point = lines[np.unravel_index(np.argmin(lines[:, :, 0]), lines.shape[0:2])]
        right_point = lines[np.unravel_index(np.argmax(lines[:, :, 0]), lines.shape[0:2])]

        left_point[0] = left_point[0] + img.shape[1] / 30
        right_point[0] = right_point[0] - img.shape[1] / 30
        
        return [Pin('1', left_point, [-1, 0]), Pin('2', right_point, [1, 0])]

class TwoPinsVert(IPinDetection):
    _COMP_TYPES = ['R_V', 'C_V', 'L_V',
    'V_V', 'A_V', 'U1', 'U2', 'I1', 'I2', 'U3', 'BAT', 'U_AC_V', 'M_V']

    NAME = 'TwoPinsVertical'
    
    @staticmethod
    def get_pins(lines: np.ndarray, img: np.ndarray):
        # rotate and then use other class
        rot_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_lines = lines[:, :, ::-1] * [1, -1]

        pins = TwoPinsHor.get_pins(rot_lines, rot_img)

        for pin in pins:
            pin.position = (pin.position * [1, -1])[::-1]
            pin.direction = (pin.direction * [1, -1])[::-1]

        return pins

class OnePinLeft(IPinDetection):
    _COMP_TYPES = ['PIN']

    NAME = 'OnePinLeft'
    
    @staticmethod
    def get_pins(lines: np.ndarray, img: np.ndarray):
        # filter vertical lines with dx and dy
        deltas = np.diff(lines, axis=-2).squeeze()
        mask = (np.diff(np.abs(deltas)) < 0).squeeze()
        lines = lines[mask, :, :]

        # get minimum y point
        left_point = lines[np.unravel_index(np.argmin(lines[:, :, 0]), lines.shape[0:2])]

        left_point[0] = left_point[0] + img.shape[1] / 30
        
        return [Pin('1', left_point, [-1, 0])]

class OnePinTop(IPinDetection):
    _COMP_TYPES = ['GND', 'GND_F', 'GND_C']

    NAME = 'OnePinTop'
    
    @staticmethod
    def get_pins(lines: np.ndarray, img: np.ndarray):
        # rotate and use other class
        rot_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_lines = lines[:, :, ::-1] * [1, -1]

        pins = OnePinLeft.get_pins(rot_lines, rot_img)

        pins[0].position = (pins[0].position * [1, -1])[::-1]
        pins[0].direction = [0, -1]

        return pins

class TwoPinsLeft(IPinDetection):
    _COMP_TYPES = ['SPK', 'MIC']

    NAME = 'TwoPinsLeft'
    
    @staticmethod
    def get_pins(lines: np.ndarray, img: np.ndarray):
        # filter vertical lines
        deltas = np.diff(lines, axis=-2).squeeze()
        mask = (np.diff(np.abs(deltas)) < 0).squeeze()
        lines = lines[mask, :, :]

        # get first point (farthest to the left)
        left_1_idx = np.unravel_index(np.argmin(lines[:, :, 0]), lines.shape[0:2])
        left_1 = lines[left_1_idx]

        # calculate verrtical bounding area around first used line with margin
        min_y, max_y = np.sort(lines[left_1_idx[0]][:, 1])
        min_y -= img.shape[1] / 50
        max_y += img.shape[1] / 50

        # remove already used line
        lines = np.delete(lines, left_1_idx[0], axis=0)

        # create mask for all lines where both endpoints are outside of the bounding rect
        mask = np.logical_or(lines[:, :, 1] < min_y , lines[:, :, 1] > max_y)
        mask = np.logical_and(mask[:, 0], mask[:, 1])
        lines = lines[mask, :, :]

        # get second point
        left_2_idx = np.unravel_index(np.argmin(lines[:, :, 0]), lines.shape[0:2])
        left_2 = lines[left_2_idx]

        left_1[0] = left_1[0] + img.shape[1] / 30
        left_2[0] = left_2[0] + img.shape[1] / 30

        return [Pin('1', left_1, [-1, 0]), Pin('2', left_2, [-1, 0])]     

ALL_DETECTORS: List[IPinDetection] = [TwoPinsHor, TwoPinsVert, OnePinLeft, OnePinTop, TwoPinsLeft]