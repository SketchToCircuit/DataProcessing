import cv2
import numpy as np
import matplotlib.pyplot as plt
import linedetection as ld

img = cv2.imread('./PinDetection/12.png', cv2.IMREAD_UNCHANGED)
if isinstance(img[0][0], np.ndarray):
    if (len(img[0][0]) == 4):
        img = 255 - img[:,:,3]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(img, kernel, 1)
kernel = np.ones((5, 5), np.uint8)
img = cv2.erode(img, kernel, 1)

lines = ld.get_lines(img, 10, 10, 2, 15, 10)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for a, b in lines:
    cv2.line(img, a, b, (0,255,0), 2)

cv2.imshow('', img)
cv2.waitKey(0)