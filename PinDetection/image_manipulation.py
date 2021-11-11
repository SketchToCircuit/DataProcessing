import cv2
import numpy as np
import math

def make_binary(img, threshold = 127):
    if isinstance(img[0][0], np.ndarray):
        if (len(img[0][0]) == 4):
            # not transparent
            if (img[0][0][3] == 255):
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = 255 - img[:, :, 3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    return img

# part should be white on black
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

# part should be white on black
def filter(img: np.ndarray):
    orig_img = img
    base_size = math.ceil(min(img.shape[0], img.shape[1]) / 40.0)

    img = cv2.blur(img, (base_size * 2 + 1, base_size * 2 + 1))
    _, img = cv2.threshold(img, 0, 355, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((base_size * 8 + 1, base_size * 8 + 1), np.uint8)
    img = cv2.dilate(img, kernel, 1)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(orig_img)
    cv2.drawContours(mask, [cnt], 0, 255, cv2.FILLED)

    return cv2.bitwise_and(orig_img, mask)