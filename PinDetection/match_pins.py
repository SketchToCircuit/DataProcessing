import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List

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
    def get_pins(lines: np.ndarray, img: np.ndarray):
        raise NotImplementedError

class TwoPinsHor(IPinDetection):
    _COMP_TYPES = ['R', 'C', 'L', 'R_H', 'C_H', 'L_H',
    'LED', 'D', 'S1', 'S2', 'BTN1', 'BTN2', 'V_H', 'A_H', 'U_AC_H',
    'LMP', 'M', 'F', 'D_Z', 'D_S', 'C_P']

    NAME = 'TwoPinsHorizontale'

    @staticmethod
    def get_pins(lines: np.ndarray, img: np.ndarray):
        left_point = lines[np.unravel_index(np.argmin(lines[:, :, 0]), lines.shape[0:2])]
        right_point = lines[np.unravel_index(np.argmax(lines[:, :, 0]), lines.shape[0:2])]
        left_point[0] = left_point[0] + img.shape[1] / 50
        right_point[0] = right_point[0] - img.shape[1] / 50
        
        return {'1': left_point, '2': right_point}

class TwoPinsVert(IPinDetection):
    _COMP_TYPES = ['R_V', 'C_V', 'L_V',
    'V_V', 'A_V', 'U1', 'U2', 'I1', 'I2', 'U3', 'BAT', 'U_AC_V', 'M_V']

    NAME = 'TwoPinsVertical'

ALL_DETECTORS: List[IPinDetection] = [TwoPinsHor, TwoPinsVert]