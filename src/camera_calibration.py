import numpy as np
import cv2
import os
from utils.helpers import resize_image

FILE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(FILE_FOLDER, "data")
CALIBRATION_IMAGE_LOCATION = os.path.join(DATA_FOLDER, "calibration_image.png")
CALIBRATED_IMAGE_OUTPUT = os.path.join(DATA_FOLDER, "calibrated_image.png")
CALIBRATION_INFO_LOCATION = os.path.join(DATA_FOLDER, "calibration_image_0_info.txt")
CALIBRATION_OUTPUT_LOCATION = os.path.join(DATA_FOLDER, "calibration_data1.npz")

def load_image(filename):
    if os.path.exists(filename):
        image = cv2.imread(filename)
        if image is None:
            print(f"Failed to load image {filename}")
        return image
    else:
        print(f"File {filename} does not exist.")
        return None

def calibrate_camera(image_path, chessboard_size):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objPoints = []
    imgPoints = []

    img = load_image(image_path)
    if img is None:
        return

    resized_img, _ = resize_image(img)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    cornersFound, cornerCoords = cv2.findChessboardCornersSB(gray, chessboard_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if cornersFound:
        objPoints.append(objp)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersRefined = cv2.cornerSubPix(gray, cornerCoords, (11, 11), (-1, -1), criteria)
        imgPoints.append(cornersRefined)

        cv2.drawChessboardCorners(resized_img, chessboard_size, cornersRefined, cornersFound)
        cv2.imwrite(CALIBRATED_IMAGE_OUTPUT, resized_img)
    else:
        print("No checkerboard found!")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    rotMat, _ = cv2.Rodrigues(rvecs[0])

    np.savez(CALIBRATION_OUTPUT_LOCATION, mtx=np.asarray(mtx, dtype=np.longdouble), 
                                    rotMat=np.asarray(rotMat, dtype=np.longdouble), 
                                    tvec=np.asarray(tvecs[0], dtype=np.longdouble), 
                                    dist=np.asarray(dist, dtype=np.longdouble))

def main():
    chessboard_size = (16, 13)
    image_path = CALIBRATION_IMAGE_LOCATION

    if os.path.exists(image_path):
        calibrate_camera(image_path, chessboard_size)
    else:
        print(f"Calibration image file {image_path} does not exist. Please capture a calibration image first.")

if __name__ == "__main__":
    main()
