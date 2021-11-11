import cv2
import numpy as np
import linedetection as ld
import image_manipulation as imgman
from match_pins import *
from typing import List
import math

img = cv2.imread('./PinDetection/testdata/SPK_3.png', cv2.IMREAD_UNCHANGED)
if img is None:
    exit()

img = imgman.make_binary(img, 100)
img = imgman.filter(img)

if img is None:
    exit()

minx, maxx, miny, maxy = imgman.get_boundary(img)
img = img[miny:maxy, minx:maxx]

kernel_size = math.ceil(min(img.shape[0], img.shape[1]) / 40.0) * 2 + 1
kernel = np.ones((kernel_size, kernel_size), np.uint8)
cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

lines = ld.get_lines(img, 10, 10, 2, 15, 10)

for detector in ALL_DETECTORS:
    if detector.match('SPK'):
        pins = [*detector.get_pins(lines, img).values()]

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pin in pins:
            cv2.circle(img, pin.position, 5, (0, 0, 255), cv2.FILLED)
            cv2.line(img, pin.position, pin.position + np.multiply(pin.direction, 10), (0, 255, 0), 2)
        cv2.imshow(detector.NAME, img)
        cv2.waitKey(0)
        break