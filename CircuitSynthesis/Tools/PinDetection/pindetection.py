import math
import os
import random
import traceback
from dataclasses import dataclass
import json

import cv2
import numpy as np
from scipy import ndimage

if __name__ == '__main__':
    from image_manipulation import *
    from linedetection import *
    from match_pins import *
else:
    from .image_manipulation import *
    from .linedetection import * 
    from .match_pins import *

@dataclass
class Component:
    component_img: np.ndarray
    '''Grayscale component image as numpy array with dimensions (h, w). Background is white.'''

    label_img: np.ndarray
    '''Grayscale label image as numpy array with dimensions (h, w). Background is white.'''

    label_offset: np.ndarray
    '''Offset of the label image (top left corner to top left corner) as 1D numpy array with size 2: [x_off, y_off]'''

    pins: Dict[str, Pin]
    '''All pins in a dictionary where the key is the name of the pin.'''

    type: str
    '''String identifier of component type'''

    _rotated: bool = False

    def rotate(self, angle):
        '''
        Rotate all pins and images by an angle in degrees. Positive angle means counter-clockwise rotation.
        This can only be done once per component, because otherwise the image would get bigger and bigger due to padding after rotation.
        Label does not get rotated.
        '''
        if self._rotated:
            raise RuntimeError("Component can only get rotated once! This is due to some reasons.")
        
        self._rotated = True

        orig_size = np.array(self.component_img.shape[::-1])
        self.component_img = ndimage.rotate(self.component_img, angle, cval=255)
        new_size = np.array(self.component_img.shape[::-1])

        self.label_offset = _rotate_point(self.label_offset - orig_size / 2 + np.array(self.label_img.shape[::-1]) / 2, -angle) + new_size / 2.0 - np.array(self.label_img.shape[::-1]) / 2

        for pin in self.pins.values():
            pin.position = _rotate_point((pin.position).astype(float) - orig_size / 2.0, -angle)
            pin.position += new_size / 2.0
            pin.direction = _rotate_point((pin.direction).astype(float), -angle)
    
    def flip(self, vert: bool = False, hor: bool = False):
        '''
        Flip the component vertically or horizontally.
        "vert" and "hor" define if you flip it vertically, horizontally or both. Default is no flip.
        Label does not get flipped.
        '''
        if hor:
            self.component_img = self.component_img[:, ::-1]
            self.label_offset[0] = self.component_img.shape[1] - self.label_offset[0] - self.label_img.shape[1]

            for pin in self.pins.values():
                pin.position[0] = self.component_img.shape[1] - pin.position[0]
                pin.direction[0] *= -1
        if vert:
            self.component_img = self.component_img[::-1, :]
            self.label_offset[1] = self.component_img.shape[0] - self.label_offset[1] - self.label_img.shape[0]

            for pin in self.pins.values():
                pin.position[1] = self.component_img.shape[0] - pin.position[1]
                pin.direction[1] *= -1
    
    def scale(self, scale_factor: float):
        '''Scale images and offsets by "scale_factor" (all axis with the same factor)'''
        self.component_img = cv2.resize(self.component_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        self.label_img = cv2.resize(self.label_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        self.label_offset = (self.label_offset.astype(float) * scale_factor).astype(int)

        for pin in self.pins.values():
            pin.position = (pin.position.astype(float) * scale_factor).astype(int)

@dataclass
class UnloadedComponent:
    '''Component where only the path to the images is stored. Use the ".load()" method to load the images and create a "Component" instance for further use.'''
    _component_path: str
    _label_path: str
    _label_offset: np.ndarray
    _pins: Dict[str, Pin]
    _type: str

    @staticmethod
    def from_dict(map):
        '''Create instance from dictionary from a *.json file.'''
        pins = {name: Pin.from_dict(pin) for (name, pin) in map['pins'].items()}
        return UnloadedComponent(map['component_path'], map['label_path'], np.array(map['label_offset']), pins, map['type'])

    def load(self):
        '''Load the images and create a "Component".'''
        cmp_img = cv2.imread(self._component_path, cv2.IMREAD_UNCHANGED)
        lbl_img = cv2.imread(self._label_path, cv2.IMREAD_UNCHANGED)

        return Component(cmp_img, lbl_img, self._label_offset, self._pins, self._type)

def import_components(data_path) -> Dict[str, List[UnloadedComponent]]:
    with open(data_path, 'r') as f:
        data_obj = json.load(f)
        data_out: Dict[str, List[UnloadedComponent]] = {}

        for type in data_obj:
            data_out[type] = []
            for cmp in data_obj[type]:
                data_out[type].append(UnloadedComponent.from_dict(cmp))

    return data_out

def combine_images(a: np.ndarray, b: np.ndarray, off):
    '''
    Overlays two images (a and b) where image b is offset by values in "off".
    off is an array with the x and y offset coordinates[x_off, y_off].
    Images are overlayed by choosing the lower values per channel.
    Resulting image is padded with white.
    Used for combining component und label image.
    '''
    off = np.round(off).astype(int)
    min_x = min(0, off[0])
    min_y = min(0, off[1])

    max_x = max(a.shape[1] - 1, b.shape[1] + off[0] - 1)
    max_y = max(a.shape[0] - 1, b.shape[0] + off[1] - 1)

    w = max_x - min_x + 1
    h = max_y - min_y + 1

    a_x = 0
    a_y = 0
    b_x = off[0]
    b_y = off[1]

    if off[0] < 0:
        a_x = -off[0]
        b_x = 0

    if off[1] < 0:
        a_y = -off[1]
        b_y = 0

    if len(a.shape) == 3 and len(b.shape) == 3:
        a = np.pad(a, ((a_y, h - a.shape[0] - a_y), (a_x, w - a.shape[1] - a_x), (0, 0)), 'constant', constant_values=255)
        b = np.pad(b, ((b_y, h - b.shape[0] - b_y), (b_x, w - b.shape[1] - b_x), (0, 0)), 'constant', constant_values=255)
    else:
        a = np.pad(a, ((a_y, h - a.shape[0] - a_y), (a_x, w - a.shape[1] - a_x)), 'constant', constant_values=255)
        b = np.pad(b, ((b_y, h - b.shape[0] - b_y), (b_x, w - b.shape[1] - b_x)), 'constant', constant_values=255)

    return np.minimum(a, b)

def _rotate_point(point, angle):
    c, s = np.cos(angle / 180 * np.pi), np.sin(angle / 180 * np.pi)
    old_point = point.copy()
    point[0] = old_point[0] * c - old_point[1] * s
    point[1] = old_point[0] * s + old_point[1] * c

    return point

def _detect_pins(path, type):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    orig_img = unify_color(img.copy())

    if img is None:
        print('Read empty image!')
        return None, None, None, None

    img = make_binary(img, 100)
    img, cropp = filter(img)
    orig_img = orig_img[cropp[0]:cropp[1], cropp[2]:cropp[3]]

    if img is None:
        print('Filtered empty image!')
        return None, None, None, None

    result = get_boundary(img)

    if result is None:
        print('No boundary!')
        return None, None, None, None

    minx, maxx, miny, maxy, centroid = result
    img = img[miny:maxy, minx:maxx]
    orig_img = orig_img[miny:maxy, minx:maxx]

    sf = 1
    if np.max(img.shape) < 100:
        sf = 200 / np.max(img.shape)
        img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
        orig_img = cv2.resize(orig_img, None, fx=sf, fy=sf, interpolation=cv2.INTER_CUBIC)

    lines = get_lines(img, 13, 10, 2, 17, 12)

    for detector in ALL_DETECTORS:
        if detector.match(type):
            try:
                pins = detector.get_pins(lines, centroid, np.array(img.shape))
                return orig_img, pins, detector.NAME, {'bounds': (minx + cropp[2], maxx + cropp[3], miny + cropp[0], maxy + cropp[1]), 'sf': sf}
            except Exception:
                print("Detection not successful: " + traceback.format_exc())
                return None, None, None, None
    
    print('No valid detector!')
    return None, None, None, None

def export_all(src_path, dest_path):
    all_json = {}

    for subdir in os.listdir(src_path):
        folder = os.path.join(src_path, subdir)
        if os.path.isdir(folder) and not folder.endswith('label'):
            files = os.listdir(folder)

            files = [os.path.join(folder, file) for file in files]
            type = os.path.split(folder)[1]

            for file in files:
                component_img = None

                try:
                    # positioning includes cropping border and scaling factor (if it did get scaled)
                    component_img, pins, name, positioning = _detect_pins(file, type)

                    if component_img is not None:
                        label_img = cv2.imread(os.path.join(folder + '_label', os.path.split(file)[1]), cv2.IMREAD_UNCHANGED)
                        label_img = unify_color(label_img)
                        label_img = cv2.resize(label_img, None, fx=positioning['sf'], fy=positioning['sf'], interpolation=cv2.INTER_AREA)
                        
                        lbl_minx, lbl_maxx, lbl_miny, lbl_maxy, _ = get_boundary(255 - label_img)
                        label_img = label_img[lbl_miny:lbl_maxy, lbl_minx:lbl_maxx]

                        label_off = np.array([-positioning['bounds'][0] * positioning['sf'] + lbl_minx, -positioning['bounds'][2] * positioning['sf'] + lbl_miny])

                        cmp = Component(component_img, label_img, label_off, pins, type)

                        if type in ['R_H', 'C_H', 'L_H']:
                            cmp.type = cmp.type[0]
                        elif type in ['R_V', 'C_V', 'L_V']:
                            cmp.rotate(90 if random.random() > 0.5 else -90)
                            cmp._rotated = False    # only allowed because it was a 90° rotation -> no padding
                            cmp.type = cmp.type[0]
                        elif type == 'POT':
                            if cmp.pins['3'].direction[1] < 0:
                                # pin direction is up -> flip vertically
                                cmp.flip(vert=True)

                except Exception:
                    component_img = None
                    print(traceback.format_exc())

                if component_img is None:
                    print('Error with file: ' + file)
                    continue

                print("Ok: " + file)

                dest_folder = os.path.join(dest_path, cmp.type)

                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

                if not os.path.exists(dest_folder + '_label'):
                    os.makedirs(dest_folder + '_label')

                cmp_path = os.path.join(dest_folder, os.path.split(file)[1])
                lbl_path = os.path.join(dest_folder + '_label', os.path.split(file)[1])
                cv2.imwrite(cmp_path, cmp.component_img)
                cv2.imwrite(lbl_path, cmp.label_img)

                if cmp.type not in all_json:
                    all_json[cmp.type] = []
                
                all_json[cmp.type].append({
                    'component_path': cmp_path,
                    'label_path': lbl_path,
                    'type': cmp.type,
                    'label_offset': cmp.label_offset.tolist(),
                    'pins': {pin_name: pin.to_dict() for (pin_name, pin) in cmp.pins.items()}
                })

    with open(os.path.join(dest_path, 'data.json'), 'w') as f:
        json.dump(all_json, f)

if __name__ == '__main__':
    export_all('../Data/data_23_11_21', './exported_data')