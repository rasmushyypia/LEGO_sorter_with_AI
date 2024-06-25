import os
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils.camera import Camera

FILE_FOLDER = os.path.dirname(__file__)
CALIBRATION_DATA = os.path.join(FILE_FOLDER, "data", "calibration_data.npz")
IMAGE_SAVE_PATH = os.path.join(FILE_FOLDER, "data", "orig_images")
CALIBRATION_INFO_FILE = os.path.join(FILE_FOLDER, "data", "calibration_image_0_info.txt")

class LegoDatasetCreator:
    def __init__(self, master, calib_data_path, init_time=100000):
        self.master = master
        self.master.title("Lego Dataset Creator")
        self.lego_id = None
        self.color = None
        self.orientation = 'a'
        self.capture_count = 0
        self.current_color = None

        self.setup_ui()

        self.cam = Camera(calib_data_path, init_time=init_time)

        # Read exposure setting from calibration info file
        exposure_time = self.read_exposure_from_calibration_info(CALIBRATION_INFO_FILE)
        if exposure_time:
            self.cam.set_exposure_time(exposure_time)

        self.update_video_feed()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.bind('<c>', self.capture_image_event)
        self.master.bind('<u>', self.capture_weird_orientation_event)
        self.master.bind('<n>', self.next_part_event)

        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

    def setup_ui(self):
        tk.Label(self.master, text="Enter Lego ID and select Color").pack()

        tk.Button(self.master, text="Set New ID (n)", command=self.set_new_part).pack()

        tk.Label(self.master, text="Select Color:").pack()
        self.color_options = ["black", "darkgrey", "grey", "beige", "white", "yellow", "orange", "red", "green", "blue"]
        self.color_var = tk.StringVar(self.master)
        self.color_var.set(self.color_options[0])
        self.color_dropdown = ttk.Combobox(self.master, textvariable=self.color_var, values=self.color_options, state='readonly')
        self.color_dropdown.pack()
        self.color_dropdown.bind("<<ComboboxSelected>>", self.on_color_selected)
        self.color_var.trace_add("write", self.color_changed)

        self.filename_label = tk.Label(self.master, text="{lego_id}_{color}_{orientation}.jpg")
        self.filename_label.pack()

        tk.Button(self.master, text="Capture Image (c)", command=self.capture_image).pack()
        tk.Button(self.master, text="Capture Weird Orientation (u)", command=self.capture_weird_orientation).pack()

        self.video_label = tk.Label(self.master)
        self.video_label.pack()

    def read_exposure_from_calibration_info(self, calibration_info_file):
        """Reads the exposure time from the calibration info file."""
        if os.path.exists(calibration_info_file):
            with open(calibration_info_file, 'r') as file:
                for line in file:
                    if "Exposure Time:" in line:
                        try:
                            exposure_time = int(line.split(":")[1].strip())
                            return exposure_time
                        except ValueError:
                            print("Invalid exposure time format in calibration info file.")
        print("Calibration info file not found or invalid format.")
        return None

    def update_filename_label(self):
        lego_id_text = self.lego_id if self.lego_id else "lego_id"
        color_text = self.color_var.get() if self.color_var.get() else "color"
        filename = f"{lego_id_text}_{color_text}_{self.orientation}.jpg"
        self.filename_label.config(text=f"Filename: {filename}")

    def set_new_part(self):
        self.lego_id = simpledialog.askstring("Input", "Enter Lego ID:")
        if self.lego_id:
            self.lego_id = self.lego_id.zfill(4)
        self.orientation = 'a'
        self.capture_count = 0
        self.current_color = self.color_var.get()
        self.update_filename_label()

    def color_changed(self, *args):
        new_color = self.color_var.get()
        if self.current_color and self.current_color != new_color:
            self.orientation = 'a'
        self.current_color = new_color
        self.update_filename_label()

    def on_color_selected(self, event):
        self.master.focus_set()

    def capture_image(self):
        if not self.validate_input():
            return

        ret, frame = self.cam.get_image()
        if ret:
            filename = os.path.join(IMAGE_SAVE_PATH, f"{self.lego_id}_{self.color}_{self.orientation}.jpg")
            cv2.imwrite(filename, frame)
            self.orientation = chr(ord(self.orientation) + 1)
            self.update_filename_label()
        else:
            messagebox.showerror("Error", "Failed to capture image")

    def capture_image_event(self, event):
        self.capture_image()

    def capture_weird_orientation(self):
        if not self.validate_input():
            return

        ret, frame = self.cam.get_image()
        if ret:
            filename = os.path.join(IMAGE_SAVE_PATH, f"{self.lego_id}_{self.color}_u.jpg")
            cv2.imwrite(filename, frame)
        else:
            messagebox.showerror("Error", "Failed to capture image")

    def capture_weird_orientation_event(self, event):
        self.capture_weird_orientation()

    def next_part_event(self, event):
        self.set_new_part()

    def validate_input(self):
        if not self.lego_id:
            messagebox.showerror("Error", "Please set Lego ID first.")
            return False
        self.color = self.color_var.get()
        if not self.color:
            messagebox.showerror("Error", "Please select a color first.")
            return False
        return True

    def update_video_feed(self):
        ret, frame = self.cam.get_image()
        if ret:
            display_frame = self.resize_image_for_display(frame, width=640, height=480)
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.master.after(10, self.update_video_feed)

    def resize_image_for_display(self, image, width, height):
        img = Image.fromarray(image)
        img_resized = img.resize((width, height), Image.ANTIALIAS)
        return np.array(img_resized)

    def on_closing(self):
        self.cam.release()
        self.master.destroy()

    def __del__(self):
        self.cam.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = LegoDatasetCreator(root, CALIBRATION_DATA)
    root.mainloop()
