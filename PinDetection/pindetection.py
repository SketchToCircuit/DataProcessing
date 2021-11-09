import cv2
import numpy as np
import linedetection as ld
import image_manipulation as imgman

img = cv2.imread('./PinDetection/10_1.png', cv2.IMREAD_UNCHANGED)
img = imgman.make_binary(img, 100)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_x = img.shape[1]
min_y = img.shape[0]
max_x = 0
max_y = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if x < min_x:
        min_x = x
    if y < min_y:
        min_y = y
    if x + w > max_x:
        max_x = x + w
    if y + h > max_y:
        max_y = y + h

img = img[min_y:max_y, min_x:max_x]

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