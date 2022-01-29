import torch
import cv2
import numpy as np
import json
from .utils import bb_intersection_over_union
from .utils import Roi
from os.path import join, dirname
import copy

# Visualization global variables
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
angle_color = (0, 255, 0)
det_color = (0, 0, 255)
other_color = (255, 0, 0)
thickness = 1

X = 0
Y = 1

# The JSON config file should be in the same folder with the detector code
FILE_PATH = dirname(__file__)
JSON_CONFIG = join(FILE_PATH, "config.json")

class Detection():

    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.label = "label_u"
        self.confidence = 0.0
        self.angle = 0
        self.best_iou = 0

    def from_yolov5_output(self, result_xyxy, names):
        self.x = int(round(result_xyxy[0].item()))
        self.y = int(round(result_xyxy[1].item()))
        self.w = int(round(result_xyxy[2].item()) - self.x)
        self.h = int(round(result_xyxy[3].item()) - self.y)
        self.label = names[int(result_xyxy[-1].item())]
        self.confidence = round(result_xyxy[-2].item(), 2)

    def from_bbox(self, bb, label, confidence):
        self.x = bb[0]
        self.y = bb[1]
        self.w = bb[2] - bb[0]
        self.h = bb[3] - bb[1]
        self.label = label
        self.confidence = confidence


    def get_bb(self):
        return [self.x, self.y, self.x+self.w, self.y+self.h]

    def get_centre(self):
        return [self.x+self.w/2, self.y+self.h/2]

    def visualize(self, image, color):
        cv2.rectangle(image, (self.x, self.y), 
                     (self.x+self.w, self.y+self.h), color, 1)

        text = "{} {}".format(self.label.split("_")[0], self.angle)
        cv2.putText(image, text, (self.x, self.y),
                    font, fontScale, color, thickness, cv2.LINE_AA)


class DetectorOutput:

    def __init__(self):
        self.label = None
        self.pick_point = (0.0, 0.0)
        self.angle = 0
        self.parts_count = 0
        self.unknown = True
        self.frame = None
        self.side = False
        self.do_shake = False

    def from_list(self, parameters):
        self.label = parameters[0]
        self.pick_point = parameters[1]
        self.angle = parameters[2]
        self.parts_count = parameters[3]
        self.unknown = parameters[4]
        self.side = parameters[5]
        self.do_shake = parameters[6]
        self.frame = parameters[7]

    def from_values(self, label, pick_point, angle, parts_count, unknown, side, do_shake, frame):
        self.label = label
        self.pick_point = pick_point
        self.angle = angle
        self.parts_count = parts_count
        self.unknown = unknown
        self.side = side
        self.do_shake = do_shake
        self.frame = frame

class Detector():

    def __init__(self, size=640, roi=[100, 100, 740, 740], angle_image_roi=[20, 45, 623, 565]):
        self.size = size
        self.roi = Roi(roi)

        self.angle_roi = Roi(angle_image_roi)

        self.x_factor = (self.roi.x2 - self.roi.x1) / size
        self.y_factor = (self.roi.y2 - self.roi.y1) / size

        self.x_offset = self.roi.x1
        self.y_offset = self.roi.y1

        self.last_centre = [0, 0]
        self.same_place_count = 0

        self.camera_calibrated = False

    def load_model(self, base_dir, model_path, confidence, iou_threshold):

        # Load the custom trained yolov5 model from local folder
        self.model = torch.hub.load(base_dir, 'custom', path=model_path, source='local')
        self.model.conf = confidence  
        self.model.iou = iou_threshold

        # Load parameters from json file
        f = open(JSON_CONFIG)
        json_dict = json.load(f)
        self.offsets = json_dict["offset"]
        self.side_pick = json_dict["side_pick"]
        self.throw = json_dict["throw"]
        self.distance = json_dict["distance"]
        self.up_edge_thresh  = json_dict["up_edge_thresh"]
        self.down_edge_thresh = json_dict["down_edge_thresh"]
        self.left_edge_thresh = json_dict["left_edge_thresh"]
        self.right_edge_thresh = json_dict["right_edge_thresh"]
        f.close()


    def detect_image(self, frame, visualize=False):

        if frame is None or len(frame) == 0:
            print("No image")
            return DetectorOutput()

        # Preprocess image
        frame = cv2.resize(frame, (self.size, self.size))

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame_gray = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If this is the first picture taken, assume its empty backlight and
        # save it as background calibration image
        if not self.camera_calibrated:
            _, self.bw_background = cv2.threshold(frame_gray, 250, 255, cv2.THRESH_BINARY)
            self.invert_bw = cv2.bitwise_not(self.bw_background)
            self.bw_background = self.bw_background / 255
            self.bw_background = self.bw_background.astype('uint8')
            self.camera_calibrated = True

        # Apply the background calibration image to the camera image to enable
        # angle detection in the edges of the light where intensity is lower
        frame_gray = np.multiply(frame_gray, self.bw_background)
        frame_gray = np.add(frame_gray, self.invert_bw)

        # Threshold the gary image for angle estimation
        _, bw = cv2.threshold(frame_gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Do inference
        results = self.model([frame], size=self.size)

        detections = []
        names = results.names
        best_score = 0.0
        index = 0

        # Save results
        for i in range(results.n):
            for j in range(len(results.xyxy[i])):
                det = Detection()
                det.from_yolov5_output(results.xyxy[i][j], names)
                if det.confidence > best_score:
                    best_score = det.confidence
                    index = j
                detections.append(det)

        # Estimate angles
        contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        angles = []
        for i, c in enumerate(contours):
            # Remove large hole contours
            if hier[0][i][3] == -1:
                continue

            # Calculate the area of each contour
            area = cv2.contourArea(c)

            # Ignore contours that are too small or too large
            if area < 800 or area > 100000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            angle = int(rect[2]) + 3
            min_rect_w = rect[1][0]
            min_rect_h = rect[1][1]

            # Retrieve the key parameters of the rotated bounding box
            angle_detection = Detection()
            angle_detection.from_bbox([int(x), int(y), 
                                       int(x + w), int(y + h)],
                                       "angle", 0.0)

            # Convert OpenCV angle to real-world angle
            if min_rect_w < min_rect_h:
                angle = 90 - angle
            else:
                angle = -angle

            # Pair the angle with a detection if possible
            for detection in detections:
                iou = bb_intersection_over_union(angle_detection.get_bb(), detection.get_bb())
                if iou > detection.best_iou:
                    detection.angle = angle
                    detection.best_iou = iou

            # Save the angle
            angle_detection.angle = angle
            angles.append(angle_detection)

        unknown = False

        # Get paired angles
        temp_angles = []
        for detection in detections:
            temp_angles.append(detection.angle)

        # Save angles that do not have a detection pair
        angles_without_dets = []
        for detection in angles:
            if detection.angle in temp_angles:
                temp_angles.pop(temp_angles.index(detection.angle))
            else:
                angles_without_dets.append(detection)

        # Determine the amount of LEGO parts in the image
        if len(angles) > len(detections):
            object_count = len(angles)
        else:
            object_count = len(detections)

        # If nothing was detected ad empty or one of the angles
        if len(detections) == 0:
            unknown = True
            if len(angles) == 0:
                detections.append(Detection())
            else:
                angles[0].label = "unknown_u"
                detections.append(angles[0])

        # Check if the part is unknown and needs to be turned around
        if detections[index].label[-1] == "u" and detections[index].label.split("_")[0] in self.throw:
            unknown = True

        label = detections[index].label.split("_")[0]
        pixel_centre = detections[index].get_centre()
        pick_point = [round(pixel_centre[X] * self.x_factor + self.x_offset, 2), 
                      round(pixel_centre[Y] * self.y_factor + self.y_offset, 2)]
        pick_angle = detections[index].angle

        # Check if the chosen parts has other parts close to it
        # if there are close parts then shake the parts in the area
        do_shake = False
        for i in range(len(detections)):
            if i == index:
                continue
            else:
                # Calculate euclidean distance to the chosen parts centre point
                centre = detections[i].get_centre()
                euclidean_distance = ((pixel_centre[X] - centre[X])**2 +
                                      (pixel_centre[Y] - centre[Y])**2) ** 0.5
                if euclidean_distance <= self.distance:
                    do_shake = True
                    break
        if not do_shake:
            for det in angles_without_dets:
                centre = det.get_centre()
                euclidean_distance = ((pixel_centre[X] - centre[X])**2 +
                                      (pixel_centre[Y] - centre[Y])**2) ** 0.5
                if euclidean_distance <= self.distance:
                    do_shake = True
                    break

        # Determine if side picking is needed
        side = False
        if label in self.side_pick:
            side = True

        # Apply needed offsets
        if label in self.offsets:
            pick_angle += self.offsets[label]

        # If the found part is too close to the edges of the detection area
        # then set it as unkonw as the detection could be invalid
        if pixel_centre[Y] <= self.up_edge_thresh or pixel_centre[Y] >= self.down_edge_thresh:
            unknown = True
        if pixel_centre[X] <= self.left_edge_thresh or pixel_centre[X] >= self.right_edge_thresh:
            unknown = True

        # Check if the current pick point is close to previous one and 
        # adjust the pick up angle if the point is close to last pick point
        euclidean_distance = ((pixel_centre[X] - self.last_centre[X])**2 +
                              (pixel_centre[Y] - self.last_centre[Y])**2) ** 0.5
        if euclidean_distance < 10:
            self.same_place_count += 1
            if self.same_place_count >= 20:
                self.same_place_count = 0
            pick_angle += 10 * self.same_place_count
        else:
            self.same_place_count += 0
            self.last_centre = copy.copy(pixel_centre)

        # Visualize detections and angles
        if visualize:
            for i in range(len(detections)):
                if i == index:
                    detections[i].visualize(frame, det_color)
                else:
                    detections[i].visualize(frame, other_color)           
            for angle_det in angles_without_dets:
                angle_det.visualize(frame, angle_color)

        # Generate output
        output = DetectorOutput()
        output.from_values(label, pick_point, pick_angle, object_count, unknown, side, do_shake, frame)
        return output


