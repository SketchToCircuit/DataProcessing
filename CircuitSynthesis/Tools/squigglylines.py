import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Lines:
    @staticmethod
    def beziercurve( points: list, step):
        pts=[]
        t = 0.0

        while t <= 1.0:
            pts.append(
                ##np.array vonverts points List into np-array 
                np.around((1-t)*(1-t)*(1-t)*np.array(points[0]) + 3*t*(1-t)*(1-t)*np.array(points[1], dtype=float) + 3*t*t*(1-t)*np.array(points[2], dtype=float) + t*t*t*np.array(points[3], dtype=float))
            )

            t += step

        return pts

    @staticmethod
    def squigglyline(x1, y1, x2, y2, picture, thickness, color):
        distance = max(math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)), 0.01)
        pointdistance = 80
        num_points = math.ceil(distance / pointdistance) + 1
        helpingpoints = []
        dx = (x2 - x1)
        dy = (y2 - y1)
        step = 0.25
        squiglines = []
        rndmborders = int(max(min(np.random.normal(5, 2), 15), 1))
         
        ##create control points
        for i in range(num_points):
            rndmlen = random.randint(-rndmborders,rndmborders)
            while rndmlen == 0:
                rndmlen = random.randint(-rndmborders,rndmborders)
            helpingpoints.append((
            x1 + dx / distance * i * pointdistance - dy / distance * rndmlen,
            y1 + dy / distance * i * pointdistance + dx / distance * rndmlen))

        helpingpoints[0] = (x1, y1)
        helpingpoints[-1] = (x2, y2)
        controlpoints = np.array(helpingpoints, dtype=np.int32)

        ##create handle points 
        for i in range(num_points):
            j = 3 * i

            randy = random.randint(-rndmborders,rndmborders)
            randx = random.randint(-rndmborders,rndmborders)

            while randy == 0 and randx == 0:
                randy = random.randint(-rndmborders,rndmborders)
                randx = random.randint(-rndmborders,rndmborders)

            helpingpoints.insert(j,(
            controlpoints[i][0] - randy,
            controlpoints[i][1] - randx
            ))

            handledistance = math.sqrt((controlpoints[i][0] - helpingpoints[j][0]) **2  + (controlpoints[i][1] - helpingpoints[j][1]) ** 2)
            rndmlen = random.randint(-rndmborders,rndmborders)
            while rndmlen == 0:
                rndmlen = random.randint(-rndmborders,rndmborders)

            helpingpoints.insert(j + 2,(
            controlpoints[i][0] + math.ceil((controlpoints[i][0] - helpingpoints[j][0]) / handledistance * rndmlen),
            controlpoints[i][1] + math.ceil((controlpoints[i][1] - helpingpoints[j][1]) / handledistance * rndmlen) 
            ))

        helpingpoints = helpingpoints[1:-1]
        handles = np.array(helpingpoints, dtype=np.int32)

        for i in range(0, len(helpingpoints) - 1, 3):
            squiglines += Lines.beziercurve(helpingpoints[i:i+4],step)
        
        line = np.array([squiglines], dtype=np.int32)

        cv2.polylines(picture, line, False, color, thickness)
       
    @staticmethod
    def linecrossing(x1, y1, x2, y2, picture, thickness, color):
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        rndmbpointlen = random.randint(int(distance * 0.25),int(distance * 0.75))
        breakpoint = (
        (x2 - x1) / distance * rndmbpointlen + x1,
        (y2 - y1) / distance * rndmbpointlen + y1)
        
        #lines need some randomly defined overlap
        overlap =random.randint(105,130) / 100

        newx2 = int((x2 - x1) / distance * rndmbpointlen * overlap + x1)
        newy2 = int((y2 - y1) / distance * rndmbpointlen * overlap + y1)

        newx1 = int((x2 - x1) / distance * rndmbpointlen * (1 / overlap) + x1) 
        newy1 = int((y2 - y1) / distance * rndmbpointlen * (1 / overlap) + y1)

        Lines.squigglyline(x1, y1, breakpoint[0], breakpoint[1], picture, thickness, (0, 0, 0))#green
        Lines.squigglyline(breakpoint[0], breakpoint[1], newx2, newy2, picture, thickness, (0, 0, 0))

        Lines.squigglyline(newx1, newy1, breakpoint[0], breakpoint[1], picture, thickness, (0, 0, 0))
        Lines.squigglyline(breakpoint[0], breakpoint[1], x2, y2, picture, thickness, (0, 0, 0))#red
        #cv2.circle(picture, np.array(np.int32([(x2 - x1) / distance * rndmbpointlen + x1, (y2 - y1) / distance * rndmbpointlen + y1])), 5, (255, 0, 255), -1)