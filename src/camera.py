from .ueyeCamera import IDSCam
import numpy as np
import cv2
from .utils import Roi

class Camera():

    def __init__(self, calib_data_path, roi=[100, 100, 740, 740], init_time = 100000):

        calib_data = np.load(calib_data_path)
        self.mtx = calib_data["mtx"]
        self.newmtx = calib_data["newmtx"]
        self.dist = calib_data["dist"]

        self.cam = IDSCam()
        self.roi = Roi(roi)
        self.cam.set_intTime(init_time)

    def get_image(self):
        image = cv2.undistort(np.asarray(self.cam.capture_image()), self.mtx, self.dist, None, self.newmtx)
        return image[self.roi.y1:self.roi.y2, self.roi.x1:self.roi.x2]


