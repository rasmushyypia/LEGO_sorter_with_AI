import numpy as np
import cv2
import os
from datetime import datetime
from utils.camera import Camera
from utils.helpers import Roi, resize_image

FILE_FOLDER = os.path.dirname(__file__)
CALIBRATION_OUTPUT_LOCATION = os.path.join(FILE_FOLDER, "data", "calibration_data.npz")
# location of calibration board taken before
CALIBRATION_IMAGE_LOCATION = os.path.join(FILE_FOLDER, "data", "calibration_image.png")

def capture_calibration_image(camera):
    # Generate a unique filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(FILE_FOLDER, "data", f"calibration_image_{timestamp}.png")
    
    # Capture a raw image for calibration
    ret, frame = camera.get_raw_image()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"Calibration image saved as {filename}")
        return filename
    else:
        print("Failed to capture calibration image.")
        return None

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
    
    # Find the chessboard corners
    cornersFound, cornerCoords = cv2.findChessboardCornersSB(gray, chessboard_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if cornersFound:
        objPoints.append(objp)
        
        # Define termination criteria for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        cornersRefined = cv2.cornerSubPix(gray, cornerCoords, (11,11), (-1,-1), criteria)
        imgPoints.append(cornersRefined)

        # Draw and display the corners
        cv2.drawChessboardCorners(resized_img, chessboard_size, cornersRefined, cornersFound)
        cv2.imshow("Calibration board", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No checkerboard found!")
        return

    # Calculate calibration matrices
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    rotMat, _ = cv2.Rodrigues(rvecs[0])

    # Save calibration results to numpy binary file
    np.savez(CALIBRATION_OUTPUT_LOCATION, mtx=np.asarray(mtx, dtype=np.longdouble), 
                                    rotMat=np.asarray(rotMat, dtype=np.longdouble), 
                                    tvec=np.asarray(tvecs[0], dtype=np.longdouble), 
                                    dist=np.asarray(dist, dtype=np.longdouble))

def main():
    mode = input("Enter mode (capture/load): ").strip().lower()
    chessboard_size = (16,13)
    
    if mode == "capture":
        camera = Camera(calib_data_path=None, init_time=50000)
        image_path = capture_calibration_image(camera)
        if image_path:
            calibrate_camera(image_path, chessboard_size)
    elif mode == "load":
        use_predefined = input(f"Do you want to use the predefined image location? (yes/no): ").strip().lower()
        if use_predefined == "yes":
            image_path = CALIBRATION_IMAGE_LOCATION
        else:
            image_path = input("Enter the path to the existing image: ").strip()
        
        if os.path.exists(image_path):
            calibrate_camera(image_path, chessboard_size)
        else:
            print(f"File {image_path} does not exist.")
    else:
        print("Invalid mode selected. Please choose either 'capture' or 'load'.")

if __name__ == "__main__":
    main()
