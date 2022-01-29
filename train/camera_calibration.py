"""
Camera calibration script
@author: Samuli Pohjola & Eetu Manninen
"""

import cv2
import os
import numpy as np
import sys

FILE_FOLDER = os.path.dirname(__file__)
# Path to calibration image needs to be given.
CALIBRATION_IMAGE = os.path.join(FILE_FOLDER, "calibration_image.png")

if __name__ == "__main__":

    # get calibartion image
    img = cv2.imread(CALIBRATION_IMAGE)

    # The rows and columns count that the OpenCV algorithm tries to find
    # These numbers worked with the original camera setup but might need to
    # be changed if camera changes. Changes in these numbers also mean changes
    # in the calibrator.py.
    rows = 6
    cols = 8

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCornersSB(gray, (rows,cols))

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
        cv2.imshow('press q to quit or any other key to continue', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            sys.exit(0)
    else:
        print("Calibration unsuccessful see the calibration guide")
        sys.exit(1)

    cv2.destroyAllWindows()

    # Calculate calibration matrixes
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    rotMat, _ = cv2.Rodrigues(rvecs[0])

    # Save calibartion results to numpy binary file
    np.savez("calibration_data.npz", mtx=np.asarray(mtx, dtype=np.longdouble), 
                                    rotMat=np.asarray(rotMat, dtype=np.longdouble), 
                                    tvec=np.asarray(tvecs[0], dtype=np.longdouble), 
                                    dist=np.asarray(dist, dtype=np.longdouble))
