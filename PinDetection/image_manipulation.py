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