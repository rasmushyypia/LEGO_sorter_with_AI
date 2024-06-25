"""
Helper functions for LEGO sorting
@author: Samuli Pohjola & Eetu Manninen
"""

import numpy as np
from numpy.linalg import inv
import cv2

class Roi:
    def __init__(self, roi, max_width, max_height):
        self.x1 = max(0, min(roi[0], max_width))
        self.y1 = max(0, min(roi[1], max_height))
        self.x2 = max(0, min(roi[2], max_width))
        self.y2 = max(0, min(roi[3], max_height))
        self.max_width = max_width
        self.max_height = max_height

    def set_roi(self, x1, y1, x2, y2):
        self.x1 = max(0, min(x1, self.max_width))
        self.y1 = max(0, min(y1, self.max_height))
        self.x2 = max(0, min(x2, self.max_width))
        self.y2 = max(0, min(y2, self.max_height))

    def get_roi(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

def resize_image(image, frame_size=(1024, 768)):
    """This function is purely used to resize image to smaller size for display purposes"""
    h, w = image.shape[:2]
    frame_w, frame_h = frame_size

    # Calculate the scale factor while maintaining the aspect ratio
    scale = min(frame_w / w, frame_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, scale

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
    # areas - the intersection area
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
