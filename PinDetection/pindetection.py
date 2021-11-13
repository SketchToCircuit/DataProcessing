import cv2
import numpy as np
import linedetection as ld
import image_manipulation as imgman
from match_pins import *
from typing import List
import math
import timeit

def main():
    #print(str(timeit.timeit(stmt="detect_pins('./PinDetection/testdata/MIC_1.png', 'MIC')", setup="from __main__ import detect_pins", number=10) / 10 * 1000) + 'ms / detection')
    #exit()

    img, pins, name = detect_pins('./PinDetection/testdata/NPN_0.png', 'NPN')

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pin in pins.values():
        cv2.circle(img, pin.position, 5, (0, 0, 255), cv2.FILLED)
        cv2.line(img, pin.position, pin.position + np.multiply(pin.direction, 10), (0, 255, 0), 2)
    cv2.imshow(name, img)
    cv2.waitKey(0)

def detect_pins(path, type):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, None

    img = imgman.make_binary(img, 100)
    img = imgman.filter(img)

    if img is None:
        return None, None, None

    result = imgman.get_boundary(img)

    if result is None:
        return None, None, None

    minx, maxx, miny, maxy, centroid = result
    img = img[miny:maxy, minx:maxx]

    kernel_size = math.ceil(min(img.shape[0], img.shape[1]) / 40.0) * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    lines = ld.get_lines(img, 10, 10, 2, 15, 10)

    for detector in ALL_DETECTORS:
        if detector.match(type):
            try:
                pins = detector.get_pins(lines, centroid, np.array(img.shape))
                return img, pins, detector.NAME
            except Exception as err:
                print("Detection not successful: " + repr(err))
                return None, None, None
    
    return None, None, None

if __name__ == '__main__':
    main()