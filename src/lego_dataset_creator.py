import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils.camera import Camera  # Ensure correct import based on folder structure
import os

FILE_FOLDER = os.path.dirname(__file__)
CALIBRATION_DATA = os.path.join(FILE_FOLDER, "data", "calibration_data.npz")
IMAGE_SAVE_PATH = os.path.join(FILE_FOLDER, "data", "orig_images")

class LegoDatasetCreator:
    def __init__(self, master, calib_data_path, init_time=100000):
        self.master = master
        self.master.title("Lego Dataset Creator")
        self.lego_id = None
        self.color = None
        self.orientation = 'a'
        self.capture_count = 0
        self.current_color = None

        self.label = tk.Label(master, text="Enter Lego ID and select Color")
        self.label.pack()

        self.btn_set_new_part = tk.Button(master, text="Set New ID (n)", command=self.set_new_part)
        self.btn_set_new_part.pack()

        self.color_label = tk.Label(master, text="Select Color:")
        self.color_label.pack()
        self.color_options = ["black", "darkgrey", "grey", "beige", "white", "yellow", "orange", "red", "green", "blue"]
        self.color_var = tk.StringVar(master)
        self.color_var.set(self.color_options[0])  # default value
        self.color_dropdown = ttk.Combobox(master, textvariable=self.color_var, values=self.color_options, state='readonly')
        self.color_dropdown.pack()
        self.color_dropdown.bind("<<ComboboxSelected>>", self.on_color_selected)

        self.color_var.trace_add("write", self.color_changed)  # Trace color changes

        self.filename_label = tk.Label(master, text="{lego_id}_{color}_{orientation}.jpg")
        self.filename_label.pack()

        self.btn_capture = tk.Button(master, text="Capture Image (c)", command=self.capture_image)
        self.btn_capture.pack()

        self.btn_capture_weird = tk.Button(master, text="Capture Weird Orientation (u)", command=self.capture_weird_orientation)
        self.btn_capture_weird.pack()

        self.video_label = tk.Label(master)
        self.video_label.pack()

        # Initialize Camera class with full resolution ROI
        full_resolution_roi = [0, 0, 2560, 1920]  # Full resolution of the camera
        self.cam = Camera(calib_data_path, full_resolution_roi, init_time)
        self.update_video_feed()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle closing event

        self.master.bind('<c>', self.capture_image_event)  # Bind 'c' key for capturing images
        self.master.bind('<u>', self.capture_weird_orientation_event)  # Bind 'u' key for capturing weird orientation images
        self.master.bind('<n>', self.next_part_event)  # Bind 'n' key for next part

        # Ensure the image save path exists
        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

    def update_filename_label(self):
        lego_id_text = self.lego_id if self.lego_id else "lego_id"
        color_text = self.color_var.get() if self.color_var.get() else "color"
        filename = f"{lego_id_text}_{color_text}_{self.orientation}.jpg"
        self.filename_label.config(text=f"Filename: {filename}")

    def set_new_part(self):
        self.lego_id = simpledialog.askstring("Input", "Enter Lego ID:")
        if self.lego_id:
            self.lego_id = self.lego_id.zfill(4)  # Zero-padding to ensure four digits
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
        # After selecting a color, move focus to the main window to avoid overwriting the color value
        self.master.focus_set()

    def capture_image(self):
        if not self.lego_id:
            messagebox.showerror("Error", "Please set Lego ID first.")
            return
        
        # Always get the latest selected color from the dropdown
        self.color = self.color_var.get()
        
        if not self.color:
            messagebox.showerror("Error", "Please select a color first.")
            return

        ret, frame = self.cam.get_image()
        if ret:
            filename = os.path.join(IMAGE_SAVE_PATH, f"{self.lego_id}_{self.color}_{self.orientation}.jpg")
            cv2.imwrite(filename, frame)

            # Increment orientation
            self.orientation = chr(ord(self.orientation) + 1)
            self.update_filename_label()
        else:
            messagebox.showerror("Error", "Failed to capture image")

    def capture_image_event(self, event):
        self.capture_image()

    def capture_weird_orientation(self):
        if not self.lego_id:
            messagebox.showerror("Error", "Please set Lego ID first.")
            return

        self.color = self.color_var.get()
        
        if not self.color:
            messagebox.showerror("Error", "Please select a color first.")
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

    def update_video_feed(self):
        ret, frame = self.cam.get_image()
        if ret:
            # Resize the frame for display
            display_frame = self.resize_image_for_display(frame, width=640, height=480)
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)  # Assuming mono camera
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
