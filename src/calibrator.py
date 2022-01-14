from .utils import groundProjectPoint
import numpy as np

class Calibrator():

    def __init__(self, calib_data_path):
        calib_data = np.load(calib_data_path)
        self.mtx = calib_data["mtx"]
        self.rotMat = calib_data["rotMat"]
        self.tvec = calib_data["tvec"]


    def project_point(self, centre_point):

        projected_point = groundProjectPoint(centre_point, self.mtx, self.rotMat, self.tvec)
        x_mm = round((projected_point[0][0]-5)*-1*10, 2)
        y_mm = round((projected_point[1][0]-1)*10, 2)
        return x_mm, y_mm


