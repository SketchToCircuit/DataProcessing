import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Lines:
    def beziercurve(self, points: list, step):
        pts=[]

        for i in range(math.floor(1/step)):
            t = i * step
            pts.append(
                np.around((1-t)*(1-t)*(1-t)*np.array(points[0]) + 3*t*(1-t)*(1-t)*np.array(points[1]) + 3*t*t*(1-t)*np.array(points[2]) + t*t*t*np.array(points[3]))
            )

        return pts


    def squigglyline(self, x1, y1, x2, y2, picture, thickness, color):
        distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
        pointdistance = 80
        num_points = math.ceil(distance / pointdistance) + 1
        helpingpoints = []
        dx = (x2 - x1)
        dy = (y2 - y1)
        step = 0.1
        squiglines = []
        rndmborders = 10
         
        ##create control points
        for i in range(num_points):
            rndmlen = random.randint(-rndmborders,rndmborders)
            helpingpoints.append((
            x1 + dx / distance * i * pointdistance - dy / distance * rndmlen,
            y1 + dy / distance * i * pointdistance + dx / distance * rndmlen))

        helpingpoints[0] = (x1, y1)
        helpingpoints[-1] = (x2, y2)
        controlpoints = np.array(helpingpoints)
        controlpoints = np.int32(controlpoints)


        ##create handle points 
        for i in range(num_points):
            j = 3 * i
            helpingpoints.insert(j,(
            controlpoints[i][0] - random.randint(-rndmborders,rndmborders),
            controlpoints[i][1] - random.randint(-rndmborders,rndmborders)
            ))

            handledistance = math.sqrt((controlpoints[i][0] - helpingpoints[j][0]) **2  + (controlpoints[i][1] - helpingpoints[j][1]) ** 2)
            rndmlen = random.randint(-rndmborders,rndmborders)
            helpingpoints.insert(j + 2,(
            controlpoints[i][0] + math.ceil((controlpoints[i][0] - helpingpoints[j][0]) / handledistance * rndmlen),
            controlpoints[i][1] + math.ceil((controlpoints[i][1] - helpingpoints[j][1]) / handledistance * rndmlen) 
            ))

        helpingpoints = helpingpoints[1: -1]
        handles = np.array(helpingpoints)
        handles = np.int32(handles)

        for i in range(0, len(helpingpoints) - 1, 3):
            squiglines += self.beziercurve(helpingpoints[i:i+4],step)
        
        line = np.array(squiglines)
        line = np.int32([line])
     
        cv2.polylines(picture, line, False, color, thickness)
        #for i in range(len(controlpoints)):
         #   cv2.circle(picture, np.array(np.int32(controlpoints[i])), 5, (255, 0, 255), -1)

        #cv2.circle(picture, np.array(np.int32([x1, y1])), 3, (0, 255, 255), -1)
        #cv2.circle(picture, np.array(np.int32([x2, y2])), 3, (0, 255, 255), -1)

    def linecrossing(self, x1, y1, x2, y2, picture, thickness, color):
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        rndmbpointlen = random.randint(int(distance * 0.25),int(distance * 0.75))
        breakpoint = (
            (x2 - x1) / distance * rndmbpointlen + x1,
            (y2 - y1) / distance * rndmbpointlen + y1
        )
        #lines need some overlap 
        overlap = random.randint(0,100)

        newx2 = (x2 - x1) / distance * rndmbpointlen * 1.2 + x1
        newy2 = (y2 - y1) / distance * rndmbpointlen * 1.2 + y1

        newx1 = (x2 - x1) / distance * rndmbpointlen * 0.9 + x1 
        newy1 = (y2 - y1) / distance * rndmbpointlen * 0.9 + y1
        print(newx2)
        print(newy2)
        print(breakpoint[0])
        print(breakpoint[1])
        
        self.squigglyline(x1, y1, newx2, newy2, picture, thickness, (0, 0, 0))#green
        self.squigglyline(newx1, newy1, x2, y2, picture, thickness, (0, 0, 0))#red
        breakpoint = np.array(breakpoint)
        breakpoint = np.int32(breakpoint)
        #cv2.circle(picture, np.array(np.int32([newx2, newy2])), 5, (255, 0, 255), -1)
        #cv2.circle(picture, breakpoint, 8, (255, 0, 0), -1)
        

def main():
    test = Lines()
    image = np.ones((512, 512, 3), np.uint8) * 255
    test.linecrossing(10, 400, 400, 10, image, 3, (0, 0, 0))

    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()