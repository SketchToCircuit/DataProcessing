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

    min_x = img.shape[1] - 1
    min_y = img.shape[0] - 1
    max_x = 0
    max_y = 0

    cum_cx = 0
    cum_cy = 0
    cum_area = 0

    # combine all bounding boxes and calculate weighted centroid
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        a = cv2.contourArea(cnt)

        cum_cx += m['m10']
        cum_cy += m['m01']
        cum_area += a

        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x + w > max_x:
            max_x = x + w
        if y + h > max_y:
            max_y = y + h

    # add padding
    min_x = max(min_x - 10, 0)
    min_y = max(min_y - 10, 0)
    max_x = min(max_x + 10, img.shape[1] - 1)
    max_y = min(max_y + 10, img.shape[0] - 1)

    if min_x > max_x:
        min_x = max_x = 0
    if min_y > max_y:
        min_y = max_y = 0

    cum_cx = cum_cx / cum_area - min_x
    cum_cy = cum_cy / cum_area - min_y
    
    return min_x, max_x, min_y, max_y, np.array([cum_cx, cum_cy], dtype=int)

# part should be white on black
def filter(img: np.ndarray):
    orig_img = img
    base_size = math.ceil(min(img.shape[0], img.shape[1]) / 30.0)

    # dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_size * 8 + 1, base_size * 8 + 1))
    img = cv2.dilate(img, kernel, 1)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)

    return orig_img[y:y+h, x:x+w]