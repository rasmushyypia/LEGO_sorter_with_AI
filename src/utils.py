"""
Utilities for LEGO sorting
@author: Samuli Pohjola & Eetu Manninen
"""

import numpy as np
from numpy.linalg import inv

class Roi():
    def __init__(self, roi):
        self.x1 = roi[0]
        self.y1 = roi[1]
        self.x2 = roi[2]
        self.y2 = roi[3]

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def CheckOverlappingBoundingBoxes(bb1, bb2):
    return not ((bb1[0]>=bb2[2]) or (bb1[2]<=bb2[0]) or (bb1[3]<=bb2[1]) or (bb1[1]>=bb2[3]))     


def groundProjectPoint(image_point, camera_matrix, rotMat, tvec, z = 0.0):
    camMat = np.asarray(camera_matrix)
    iRot = inv(rotMat)
    iCam = inv(camMat)

    uvPoint = np.ones((3, 1))

    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))

    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z

    return wcPoint