import math
import os
import random
import timeit
import traceback
from dataclasses import dataclass
import json

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.measurements import label

import image_manipulation as imgman
import linedetection as ld
from match_pins import *

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

    __rotated: bool = False

    def rotate(self, angle):
        '''
        Rotate all pins and images by an angle in degrees. Positive angle means counter-clockwise rotation.
        This can only be done once per component, because otherwise the image would get bigger and bigger due to padding after rotation.
        Label does not get rotated.
        '''
        if self.__rotated:
            raise RuntimeError("Component can only get rotated once! This is due to some reasons.")
        
        self.__rotated = True

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

    a = np.pad(a, ((a_y, h - a.shape[0] - a_y), (a_x, w - a.shape[1] - a_x), (0, 0)), 'constant', constant_values=255)
    b = np.pad(b, ((b_y, h - b.shape[0] - b_y), (b_x, w - b.shape[1] - b_x), (0, 0)), 'constant', constant_values=255)

    return np.minimum(a, b)

def _rotate_point(point, angle):
    c, s = np.cos(angle / 180 * np.pi), np.sin(angle / 180 * np.pi)
    old_point = point.copy()
    point[0] = old_point[0] * c - old_point[1] * s
    point[1] = old_point[0] * s + old_point[1] * c

    return point

def _test_files():
    data_dir = '../Data/data_9_11/saved'
    for subdir in os.listdir(data_dir)[::-1]:
        folder = os.path.join(data_dir, subdir)
        if os.path.isdir(folder) and not folder.endswith('label'):
            files = os.listdir(folder)
            files = random.sample(files, k=1)
            files = [os.path.join(folder, file) for file in files]
            type = os.path.split(folder)[1]

            for file in files:
                img, pins, name, positioning = _detect_pins(file, type)
                if img is None:
                    print('Error with file: ' + file)
                    continue

                print("Ok: " + file)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for pin in pins.values():
                    cv2.circle(img, pin.position, 5, (0, 0, 255), cv2.FILLED)
                    cv2.line(img, pin.position, pin.position + np.multiply(pin.direction, 10), (0, 255, 0), 2)
                cv2.imshow(file, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

def _main():
    export_all('../Data/data_9_11/saved', '')
    exit()
    #_test_files()
    #exit()
    #print(str(timeit.timeit(stmt="detect_pins('./PinDetection/testdata/MIC_1.png', 'MIC')", setup="from __main__ import detect_pins", number=10) / 10 * 1000) + 'ms / detection')
    #exit()
    return

def _detect_pins(path, type):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    orig_img = imgman.unify_color(img.copy())

    if img is None:
        print('Read empty image!')
        return None, None, None, None

    img = imgman.make_binary(img, 100)
    img, cropp = imgman.filter(img)
    orig_img = orig_img[cropp[0]:cropp[1], cropp[2]:cropp[3]]

    if img is None:
        print('Filtered empty image!')
        return None, None, None, None

    result = imgman.get_boundary(img)

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
        print(sf)

    kernel_size = math.ceil(min(img.shape[0], img.shape[1]) / 40.0) * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    lines = ld.get_lines(img, 10, 10, 2, 15, 10)

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
    for subdir in os.listdir(src_path):
        folder = os.path.join(src_path, subdir)
        if os.path.isdir(folder) and not folder.endswith('label'):
            files = os.listdir(folder)
            files = random.sample(files, k=1)
            files = [os.path.join(folder, file) for file in files]
            type = os.path.split(folder)[1]

            for file in files:
                component_img = None

                try:
                    # positioning includes cropping border and scaling factor (if it did get scaled)
                    component_img, pins, name, positioning = _detect_pins(file, type)

                    if component_img is not None:
                        label_img = cv2.imread(os.path.join(folder + '_label', os.path.split(file)[1]), cv2.IMREAD_UNCHANGED)
                        label_img = imgman.unify_color(label_img)
                        label_img = cv2.resize(label_img, None, fx=positioning['sf'], fy=positioning['sf'], interpolation=cv2.INTER_AREA)
                        
                        lbl_minx, lbl_maxx, lbl_miny, lbl_maxy, _ = imgman.get_boundary(255 - label_img)
                        label_img = label_img[lbl_miny:lbl_maxy, lbl_minx:lbl_maxx]

                        label_off = np.array([-positioning['bounds'][0] * positioning['sf'] + lbl_minx, -positioning['bounds'][2] * positioning['sf'] + lbl_miny])

                        cmp = Component(component_img, label_img, label_off, pins)
                        #cmp.rotate(45)
                        cmp.flip(vert=True, hor=True)
                        cmp.rotate(-180)
                        # TODO: export
                except Exception:
                    component_img = None
                    print(traceback.format_exc())

                if component_img is None:
                    print('Error with file: ' + file)
                    continue

                print("Ok: " + file)

                component_img = cv2.cvtColor(cmp.component_img, cv2.COLOR_GRAY2BGR)
                label_img = cv2.cvtColor(cmp.label_img, cv2.COLOR_GRAY2BGR)

                for pin in cmp.pins.values():
                    cv2.circle(component_img, pin.position.astype(int), 5, (0, 0, 255), cv2.FILLED)
                    cv2.line(component_img, pin.position.astype(int), (pin.position + np.multiply(pin.direction, 10)).astype(int), (0, 255, 0), 2)

                cv2.imshow(file, combine_images(component_img, label_img, cmp.label_offset))
                #cv2.imshow(file, label_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == '__main__':
    _main()
