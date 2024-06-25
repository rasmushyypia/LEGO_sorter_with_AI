import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import simpledialog
from utils.camera import Camera
from utils.helpers import resize_image

FILE_FOLDER = os.path.dirname(__file__)
CALIBRATION_INPUT_LOCATION = os.path.join(FILE_FOLDER, "data", "calibration_data.npz")
BACKGROUND_IMAGE_LOCATION = os.path.join(FILE_FOLDER, "data", "background_image.png")

def nothing(x):
    pass

# Global variables for ROI
roi_start_x = 100
roi_start_y = 100
roi_size = 100
dragging = False
capture_roi = False
scale_factor = 1.0

def draw_rectangle(event, x, y, flags, param):
    global roi_start_x, roi_start_y, roi_size, dragging, capture_roi, scale_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        roi_start_x = int(x / scale_factor)
        roi_start_y = int(y / scale_factor)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            roi_start_x = int(x / scale_factor) - roi_size // 2
            roi_start_y = int(y / scale_factor) - roi_size // 2

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def capture_image():
    global capture_roi
    capture_roi = True

def update_exposure(val):
    exposure_time = int(val)
    camera.set_exposure_time(exposure_time)
    exposure_entry.delete(0, tk.END)
    exposure_entry.insert(0, str(exposure_time))

def update_roi_size(val):
    global roi_size
    roi_size = int(val)
    roi_entry.delete(0, tk.END)
    roi_entry.insert(0, str(roi_size))

def set_exposure():
    exposure_time = int(exposure_entry.get())
    exposure_scale.set(exposure_time)
    camera.set_exposure_time(exposure_time)

def set_roi_size():
    global roi_size
    roi_size = int(roi_entry.get())
    roi_scale.set(roi_size)

def main():
    global roi_start_x, roi_start_y, roi_size, capture_roi, scale_factor, camera, exposure_entry, roi_entry, exposure_scale, roi_scale

    camera = Camera(calib_data_path=CALIBRATION_INPUT_LOCATION)

    root = tk.Tk()
    root.title("Camera Control")

    exposure_frame = tk.Frame(root)
    exposure_frame.pack(fill='x')
    exposure_label = tk.Label(exposure_frame, text="Exposure")
    exposure_label.pack(side='left')
    exposure_entry = tk.Entry(exposure_frame)
    exposure_entry.pack(side='left')
    exposure_entry.insert(0, '15000')
    exposure_button = tk.Button(exposure_frame, text="Set", command=set_exposure)
    exposure_button.pack(side='left')

    exposure_scale = tk.Scale(root, from_=0, to=250000, orient='horizontal', command=update_exposure)
    exposure_scale.set(15000)
    exposure_scale.pack(fill='x')

    roi_frame = tk.Frame(root)
    roi_frame.pack(fill='x')
    roi_label = tk.Label(roi_frame, text="ROI Size")
    roi_label.pack(side='left')
    roi_entry = tk.Entry(roi_frame)
    roi_entry.pack(side='left')
    roi_entry.insert(0, '100')
    roi_button = tk.Button(roi_frame, text="Set", command=set_roi_size)
    roi_button.pack(side='left')

    roi_scale = tk.Scale(root, from_=50, to=2000, orient='horizontal', command=update_roi_size)
    roi_scale.set(100)
    roi_scale.pack(fill='x')

    capture_button = tk.Button(root, text="Capture ROI", command=capture_image)
    capture_button.pack(fill='x')

    cv2.namedWindow('Camera Feed')
    cv2.setMouseCallback('Camera Feed', draw_rectangle)

    while True:
        ret, image = camera.get_raw_image()

        # Convert the image to BGR if it is grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Resize the image for display
        image_display, scale_factor = resize_image(image)

        # Draw the ROI rectangle on the resized image
        h_ratio = image_display.shape[0] / image.shape[0]
        w_ratio = image_display.shape[1] / image.shape[1]
        roi_display_x = int(roi_start_x * w_ratio)
        roi_display_y = int(roi_start_y * h_ratio)
        roi_display_size = int(roi_size * min(h_ratio, w_ratio))

        # Change the color of the bounding box here
        bounding_box_color = (0, 0, 255)  # Red in BGR format
        cv2.rectangle(image_display, (roi_display_x, roi_display_y), 
                      (roi_display_x + roi_display_size, roi_display_y + roi_display_size), 
                      bounding_box_color, 2)

        # Display ROI values on the image
        roi_text = f'ROI: [{roi_start_x}, {roi_start_y}, {roi_start_x + roi_size}, {roi_start_y + roi_size}]'
        cv2.putText(image_display, roi_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the image with ROI
        cv2.imshow('Camera Feed', image_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if capture_roi:
            # Capture the ROI area
            roi_image = image[roi_start_y:roi_start_y + roi_size, roi_start_x:roi_start_x + roi_size]
            cv2.imshow('Captured ROI', roi_image)
            cv2.imwrite('background_image.png', roi_image)
            capture_roi = False

        root.update_idletasks()
        root.update()

    camera.release()
    cv2.destroyAllWindows()
    root.destroy()

if __name__ == "__main__":
    main()
