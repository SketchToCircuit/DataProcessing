import random
import numpy as np
import cv2

ERODE_PROB = 0.8
MAX_ERODE_SIZE = 3

def augment_cmp_img(img):
    return img