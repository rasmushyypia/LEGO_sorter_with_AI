"""
Calibration calculation class
@author: Samuli Pohjola & Eetu Manninen
"""

from .utils import groundProjectPoint
import numpy as np

class Calibrator():

    def __init__(self, calib_data_path):
        # Loads calibartion data from the numpy file
        calib_data = np.load(calib_data_path)
        self.mtx = calib_data["mtx"]
        self.rotMat = calib_data["rotMat"]
        self.tvec = calib_data["tvec"]

        # This number is determined by the lower edge of the detected
        # chessboard in calibration_test.py. Its the distance in chessboard
        # tiles from the lower edge to the origo of the backlight.
        self.calibration_x_offset = -5

        # Same for this number but it is the distance in chessboard
        # tiles from the left edge to the origo of the backlight.
        self.calibration_y_offset = -1

        # Calibration chessboard tile size
        self.chessboard_tile_size = 10


    def project_point(self, centre_point):

        # Project pixel coordinates to millimeter coordinates according to the
        # camera calibration data.
        projected_point = groundProjectPoint(centre_point, self.mtx, self.rotMat, self.tvec)
        x_mm = round((projected_point[0][0]+self.calibration_x_offset)*-1*self.chessboard_tile_size, 2)
        y_mm = round((projected_point[1][0]+self.calibration_y_offset)*self.chessboard_tile_size, 2)
        return x_mm, y_mm


