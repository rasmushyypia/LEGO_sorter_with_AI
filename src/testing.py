import cv2
import os
import tkinter as tk
from tkinter import simpledialog
from utils.camera import Camera
from utils.helpers import resize_image

FILE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(FILE_FOLDER, "data")

# Ensure the data directory exists
os.makedirs(DATA_FOLDER, exist_ok=True)

CALIBRATION_INPUT_LOCATION = os.path.join(DATA_FOLDER, "calibration_data.npz")
BACKGROUND_IMAGE_LOCATION = os.path.join(DATA_FOLDER, "background_image.png")
CALIBRATION_IMAGE_LOCATION = os.path.join(DATA_FOLDER, "calibration_image.png")

# Global variables for ROI
roi_start_x = 100
roi_start_y = 100
roi_size = 100
dragging = False
capture_roi = False
capture_calibration = False
use_calibration = False
scale_factor = 1.0

def nothing(x):
    """A dummy callback function."""
    pass

def draw_rectangle(event, x, y, flags, param):
    """Handles mouse events for drawing and updating the ROI rectangle."""
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

def capture_roi_image():
    """Sets the flag to capture the ROI."""
    global capture_roi
    capture_roi = True

def capture_calibration_image():
    """Sets the flag to capture raw image for calibration."""
    global capture_calibration
    capture_calibration = True

def update_exposure(val):
    """Updates the camera exposure time based on the slider value."""
    global camera
    exposure_time = int(val)
    camera.set_exposure_time(exposure_time)
    exposure_entry.delete(0, tk.END)
    exposure_entry.insert(0, str(exposure_time))

def update_roi_size(val):
    """Updates the ROI size based on the slider value."""
    global roi_size
    roi_size = int(val)
    roi_entry.delete(0, tk.END)
    roi_entry.insert(0, str(roi_size))

def set_exposure():
    """Sets the camera exposure time from the entry field value and updates the slider."""
    global camera
    exposure_time = int(exposure_entry.get())
    exposure_scale.set(exposure_time)
    camera.set_exposure_time(exposure_time)

def set_roi_size():
    """Sets the ROI size from the entry field value and updates the slider."""
    global roi_size
    roi_size = int(roi_entry.get())
    roi_scale.set(roi_size)

def setup_gui(root):
    """Sets up the GUI components for camera control."""
    global exposure_entry, roi_entry, exposure_scale, roi_scale

    root.title("Camera Control")

    # Exposure Controls
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
    exposure_scale.set(85000)
    exposure_scale.pack(fill='x')

    # ROI Controls
    roi_frame = tk.Frame(root)
    roi_frame.pack(fill='x')
    roi_label = tk.Label(roi_frame, text="ROI Size")
    roi_label.pack(side='left')
    roi_entry = tk.Entry(roi_frame)
    roi_entry.pack(side='left')
    roi_entry.insert(0, '100')
    roi_button = tk.Button(roi_frame, text="Set", command=set_roi_size)
    roi_button.pack(side='left')

    roi_scale = tk.Scale(root, from_=50, to=2200, orient='horizontal', command=update_roi_size)
    roi_scale.set(100)
    roi_scale.pack(fill='x')

    capture_button = tk.Button(root, text="Capture ROI", command=capture_roi_image)
    capture_button.pack(fill='x')
    calibration_button = tk.Button(root, text="Capture Full Image", command=capture_calibration_image)
    calibration_button.pack(fill='x')

def save_camera_info(folder, base_filename, roi, exposure_time_str):
    """Saves camera information (ROI and exposure time) to a text file."""
    info_filename = f"{base_filename}_info.txt"
    file_path = os.path.join(folder, info_filename)
    with open(file_path, 'w') as file:
        file.write(f"ROI: {roi}\n")
        file.write(f"Exposure Time: {exposure_time_str}\n")
    print(f"Camera info saved as {file_path}")

def save_incrementing_image(folder, base_filename, image, roi, exposure_time_str):
    """Saves an image with an incrementing filename to avoid overwriting existing files and saves camera info."""
    i = 0
    while os.path.exists(os.path.join(folder, f"{base_filename}_{i}.png")):
        i += 1
    incremented_filename = f"{base_filename}_{i}"
    image_filename = f"{incremented_filename}.png"
    file_path = os.path.join(folder, image_filename)
    cv2.imwrite(file_path, image)
    print(f"Image saved as {file_path}")
    save_camera_info(folder, incremented_filename, roi, exposure_time_str)

def process_and_display_image(image):
    """Processes and displays the image with the ROI."""
    global roi_start_x, roi_start_y, roi_size, capture_roi, capture_calibration, scale_factor

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

    bounding_box_color = (0, 0, 255)  # Red in BGR format
    cv2.rectangle(image_display, (roi_display_x, roi_display_y),
                  (roi_display_x + roi_display_size, roi_display_y + roi_display_size),
                  bounding_box_color, 2)

    # Display ROI values on the image
    roi_text = f'ROI: [{roi_start_x}, {roi_start_y}, {roi_start_x + roi_size}, {roi_start_y + roi_size}]'
    cv2.putText(image_display, roi_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the image with ROI
    cv2.imshow('Camera Feed', image_display)

    roi = (roi_start_x, roi_start_y, roi_start_x + roi_size, roi_start_y + roi_size)
    exposure_time_str = exposure_entry.get()  # Retrieve the exposure time from the GUI entry field

    if capture_roi:
        # Capture the ROI area
        roi_image = image[roi_start_y:roi_start_y + roi_size, roi_start_x:roi_start_x + roi_size]
        save_incrementing_image(DATA_FOLDER, 'background_image', roi_image, roi, exposure_time_str)
        capture_roi = False

    if capture_calibration:
        save_incrementing_image(DATA_FOLDER, 'calibration_image', image, roi, exposure_time_str)
        capture_calibration = False

def main_loop(camera, root):
    """Main loop to capture and display images from the camera, handle user inputs, and update the GUI."""
    try:
        while True:
            if use_calibration:
                ret, image = camera.get_image()
            else:
                ret, image = camera.get_raw_image()

            if not ret:
                print("Failed to capture image")
                break

            process_and_display_image(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            root.update_idletasks()
            root.update()
    except Exception as e:
        print(f"Error during main loop: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        root.destroy()

def main():
    global camera, use_calibration

    root = tk.Tk()
    response = simpledialog.askstring("Calibration", "Do you want to use the calibration file? (yes/no)")
    use_calibration = response.lower() in ['yes', 'y']
    calib_data_path = CALIBRATION_INPUT_LOCATION if use_calibration else None

    camera = Camera(calib_data_path=calib_data_path)

    setup_gui(root)

    cv2.namedWindow('Camera Feed')
    cv2.setMouseCallback('Camera Feed', draw_rectangle)

    main_loop(camera, root)

if __name__ == "__main__":
    main()
