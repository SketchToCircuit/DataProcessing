import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

class Lines:
    def squigglyline(self, x1, y1, x2, y2, picture):
        pts = np.array(abs(x2 - x1))

        for i in pts:



        cv2.polylines()

        return picture
    
    def linecrossing(self, x, y, picture):

    

def main():
    print("test")
    ##img = cv2.imread('./CircuitSynthesis/Test Pictures/')
    testlines = Lines
    blank_image = np.zeros((256,512,3), np.uint8)
    testlines.squigglyline(50, 200, blank_image)
    cv2.imshow('lines weee', testlines.squigglyline(50, 200, blank_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

