import random
import numpy as np
import cv2

ERODE_PROB = 0.8
MAX_ERODE_SIZE = 3

def augment_cmp_img(img):
    if random.random() < 0.5:
        img = cv2.erode(img, np.ones((random.randint(1, MAX_ERODE_SIZE), random.randint(1, MAX_ERODE_SIZE))))
    
    return img