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
        num_points = math.floor(distance / pointdistance) + 1
        helpingpoints = []
        dx = (x2 - x1)
        dy = (y2 - y1)
        step = 0.1
        squiglines = []
        rndmborders = 7
         
        ##create control points
        for i in range(num_points):
            rndmlen = random.randint(-rndmborders,rndmborders)
            helpingpoints.append((
            x1 + dx / distance * i * pointdistance - dy / distance * rndmlen,
            y1 + dy / distance * i * pointdistance + dx / distance * rndmlen))

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

    def linecrossing(self, x1, y1, x2, y2, picture, thickness, color):
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        rndmbpointlen = random.randint(int(distance * 0.25),int(distance * 0.75))
        breakpoint = (
            abs(x2-x1) / distance * rndmbpointlen,
            abs(y2 - y1) / distance * rndmbpointlen
        )
        #lines need some overlap and some perpendicular distance
        overlap = random.randint(0,20)
        perpdist = random.randint(-5,5)

        newx2 = int(breakpoint[0] * ((rndmbpointlen + overlap) / rndmbpointlen))
        newy2 = int(breakpoint[1] * ((rndmbpointlen + perpdist)  / rndmbpointlen))

        newx1 = int(breakpoint[0] * ((abs(distance - rndmbpointlen) + overlap)  / rndmbpointlen))
        newy1 = int(breakpoint[1] * ((abs(distance - rndmbpointlen) - perpdist)  / rndmbpointlen))
        print("red...........")
        print(newx1)
        print(newy1)
        print(x2)
        print(y2)
        print("green......")
        print(x1)
        print(y1)
        print(newx2)
        print(newy2)

        self.squigglyline(x1, y1, newx2, newy2, picture, thickness, (0, 255, 0))#green
        self.squigglyline(newx1, newy1, x2, y2, picture, thickness, (0, 0, 255))#red
        

def main():
    print("Hello World!")
    test = Lines()
    image = np.ones((512, 512, 3), np.uint8) * 255
    test.linecrossing(10, 400, 400, 10, image, 3, (0, 0, 0))

    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()