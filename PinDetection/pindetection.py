import cv2
import numpy as np
import linedetection as ld
import image_manipulation as imgman
from match_pins import *
from typing import List

img = cv2.imread('./PinDetection/R_14.png', cv2.IMREAD_UNCHANGED)
img = imgman.make_binary(img, 100)

minx, maxx, miny, maxy = imgman.get_boundary(img)

img = img[miny:maxy, minx:maxx]

kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(img, kernel, 1)
kernel = np.ones((5, 5), np.uint8)
img = cv2.erode(img, kernel, 1)

lines = np.array(ld.get_lines(img, 10, 10, 2, 15, 10))

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#for a, b in lines:
#    cv2.line(img, a, b, (0,255,0), 2)

for detector in ALL_DETECTORS:
    if detector.match('R'):
        pins = detector.get_pins(lines, img)
        cv2.circle(img, pins['1'], 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, pins['2'], 5, (0, 0, 255), cv2.FILLED)
        cv2.imshow(detector.NAME, img)
        cv2.waitKey(0)
        break