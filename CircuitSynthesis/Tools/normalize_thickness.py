import numpy as np
import math
import cv2

def get_line_thic(img, line):
    thic = []
    NUM_L = 10
    for i in range(NUM_L):
        vect = line[0] - line[1]
        normal = vect[::-1] * [-1, 1]
        normal = normal / np.sqrt(np.sum(np.square(normal))) * 30
        mid = line[0] + (line[1] - line[0]) / (NUM_L + 1) * (i+1)

        n_line = [mid - normal / 2.0, mid + normal / 2.0]
        n_line = np.where(n_line >= np.array(img.shape, dtype=int)[::-1], np.array(img.shape, dtype=int)[::-1] - 1, n_line)
        n_line = np.where(n_line < 0, 0, n_line)

        x = np.linspace(n_line[0][0], n_line[1][0], 30, dtype=int)
        y = np.linspace(n_line[0][1], n_line[1][1], 30, dtype=int)

        p = img[y, x]

        thic.append(np.count_nonzero(p < 128))

    thic = np.sort(thic)
    return math.floor(np.mean(thic[math.floor(len(thic) / 5.0):math.ceil(len(thic) / 3.0)]))

def normalize_thic(img, img_perc):
    lines = cv2.HoughLinesP(255 - img, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=50).squeeze()
    lines = np.reshape(lines, (lines.shape[0], 2, 2))

    thic = []

    for i in range(6):
        max_i = np.argmax(np.sqrt(np.sum(np.square(lines))))
        line = lines[max_i]
        lines = np.delete(lines, max_i, axis=0)
        thic.append(get_line_thic(img, line))
    
    thickness = np.min(thic) * 0.8 + np.mean(thic) * 0.1 + np.median(thic) * 0.1
    goal = round(max(np.max(img.shape[:2]) * img_perc, 2))
    delta = int(goal - thickness)

    if delta > 0:
        img = cv2.erode(img, np.ones((delta, delta), dtype=int), borderType=cv2.BORDER_CONSTANT, borderValue=255)
    elif delta < 0:
        img = cv2.dilate(img, np.ones((int(-delta / 2), int(-delta / 2)), dtype=int), borderType=cv2.BORDER_CONSTANT, borderValue=255)