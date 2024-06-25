from .ueye_camera import IDSCam
import numpy as np
import cv2
from .helpers import Roi
import os

FILE_FOLDER = os.path.dirname(__file__)
CALIBRATION_OUTPUT_LOCATION = os.path.join(FILE_FOLDER, "calibration_data.npz")

class Camera:
    def __init__(self, calib_data_path=None, roi=None, init_time=100000):
        self.calibrated = False

        # Check if calibration data file exists
        if calib_data_path and os.path.exists(calib_data_path):
            # Load calibration data from the numpy file
            calib_data = np.load(calib_data_path)
            self.mtx = calib_data["mtx"]
            self.dist = calib_data["dist"]
            self.calibrated = True
        else:
            print("Calibration data not found. Proceeding without calibration.")

        # Initialize the camera and capture a frame to get full resolution
        self.cam = IDSCam()
        full_res_image = np.asarray(self.cam.capture_image())
        height, width = full_res_image.shape[:2]

        # Set ROI to full resolution by default, or use provided ROI
        if roi is None:
            roi = [0, 0, width, height]

        self.roi = Roi(roi, width, height)
        self.cam.set_intTime(init_time)

    def get_image(self):
        # Capture image
        frame = np.asarray(self.cam.capture_image())
        if self.calibrated:
            # Undistort if calibration data is available
            image = cv2.undistort(frame, self.mtx, self.dist, None)
            return True, image[self.roi.y1:self.roi.y2, self.roi.x1:self.roi.x2]
        else:
            # Return raw image if not calibrated with a warning
            print("Warning: The image is not undistorted because the camera is not calibrated.")
            return True, frame[self.roi.y1:self.roi.y2, self.roi.x1:self.roi.x2]

    def get_raw_image(self):
        # Capture raw image without undistortion
        frame = np.asarray(self.cam.capture_image())
        return True, frame[self.roi.y1:self.roi.y2, self.roi.x1:self.roi.x2]

    def set_exposure_time(self, time):
        self.cam.set_intTime(time)

    def set_roi(self, x1, y1, x2, y2):
        # Set a new ROI
        self.roi.set_roi(x1, y1, x2, y2)

    def release(self):
        self.cam.disconnect()
