import cv2
import numpy as np

def make_binary(img, threshold = 127):
    if isinstance(img[0][0], np.ndarray):
        if (len(img[0][0]) == 4):
            img = 255 - img[:,:,3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    return img

def get_boundary(img):
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
    
    if (min_x > max_x):
        min_x = max_x = 0
    if (min_y > max_y):
        min_y = max_y = 0

    return min_x, max_x, min_y, max_y