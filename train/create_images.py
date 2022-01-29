"""
Training image generator script
@author: Samuli Pohjola & Eetu Manninen
"""

import numpy as np
import cv2
import os
from scipy import ndimage
import multiprocessing as mp

IMAGE_SIZE = (1920, 1920)
OUT_IMAGE_SIZE = (640, 640)
# How many parts can be in one training or validation image
MAX_NUMBER_OF_OBJECTS = 10
NUMBER_OF_IMAGES = 10000
NUMBER_OF_PROCESSES = 10
# TRAIN = True creates training images, TRAIN = False creates validation images.
# Both are needed for the training process
TRAIN = True
# This variable determines how many occurances more the most reoccuring part can
# have in comparinson to the least reoccuring part in the created images. Best
# practise is to keep this value quite small.
K = 3

# This list is used to convert labels to indeces used by the Yolo format 
# annotation data text files. It has to contain all possible training classes
# in the same order as they are stated in the legoset.yaml so copying it from 
# the legoset.yaml is recommended.
LABELS = [
          '1_a', '1_u', '2_a', '2_u','4_a', '4_u', '5_a', '5_u', '6_a', '6_u', '7_a', '7_u', '11_a', '11_u', '20_a', '20_b',
          '21_a', '21_b', '22_a', '22_b',
          '30_a', '30_u', '31_a', '31_u', '33_a', '33_b', '34_a', '34_b', '35_a', '35_b', '36_a', '36_b', '37_a', '37_b',
          '40_a', '41_a','41_b', '41_u','61_a','63_a','69_a','70_a', '70_b','71_a', '90_a', '90_u', '91_a', '91_b', 
          '92_a', '92_u','93_a', '93_b', '93_c', '94_a', '94_b', '94_c', '94_d',
          '101_a', '102_a', '103_a', '104_a', '105_a','106_a', '108_a','120_a', '141_a', '142_a','180_a', '180_b', 
          '181_a', '190_a', '190_b', '192_a', '192_b', '194_a', '194_b',
          '200_a', '200_b', '201_a', '201_b', '202_a', '202_b', '204_a', '204_b', '230_a', '230_b', '231_a', '232_a',
          '900_a',  '901_a', '902_a', '902_b', '903_a', '903_b',
          '10_a', '10_b', '60_a', '60_b', '62_a', '62_b', '95_a', '143_a', '144_a', '250_a', '250_b',
          '250_c', '252_a', '253_a', '38_a', '38_b', '38_c', '96_a', '96_b', '96_u', '109_a', '256_a',
          '256_b', '259_a', '260_a', '42_a', '254_a', '254_b', '110_a', '43_a', '43_b', '43_c', '43_d', 
          '43_e', '108_a', '181_b', '233_a', '233_b', '251_a', '270_a', '21_c'
        ]

FILE_FOLDER = os.path.dirname(__file__)
TEMPLATE_FOLDER = os.path.join(FILE_FOLDER, "templates")
TRAIN_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "dataset", "images", "train")
TRAIN_LABELS_FOLDER = os.path.join(FILE_FOLDER, "dataset", "labels", "train")

VAL_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "dataset", "images", "val")
VAL_LABELS_FOLDER = os.path.join(FILE_FOLDER, "dataset", "labels", "val")

# The background image might need to be changes if the camera or the backlight is changed
# The background image should have an aspect ratio of 1:1
TEMPLATE_IMAGE = os.path.join(FILE_FOLDER, "background.png")

def GetTemplates():
    files = os.listdir(TEMPLATE_FOLDER)

    fileList = []

    for file in files:
        if file.endswith(".png"):
            fileList.append(file)

    return fileList

def GetTemplateImage(name):
    image = cv2.imread(os.path.join(TEMPLATE_FOLDER, name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def CheckOverlappingGrayAreas(image, template, bb):
    _, mask = cv2.threshold(template, 225, 255, cv2.THRESH_BINARY)
    mask = mask / 255
    mask = mask.astype('uint8')
    _, mask2 = cv2.threshold(image[bb[1]:bb[3], bb[0]:bb[2]], 225, 255, cv2.THRESH_BINARY)
    mask2 = mask2 / 255
    mask2 = mask2.astype('uint8')

    # Check if gray areas overlap in over 3 pixels
    overlaps = np.sum(np.multiply(mask, mask2)) > 3
    return overlaps

def CheckOverlappingBoundingBoxes(bb1, bb2):
    return not ((bb1[0]>=bb2[2]) or (bb1[2]<=bb2[0]) or (bb1[3]<=bb2[1]) or (bb1[1]>=bb2[3]))

def BoundingBoxesOverlap(bb, boundingBoxes, image, template):
    if len(boundingBoxes) > 0:
        for b in boundingBoxes:
            if CheckOverlappingBoundingBoxes(bb, b):
                if CheckOverlappingGrayAreas(image, template, bb):
                    return True
    return False

def GetBoundingBox(template, templateName):
    rng = np.random.default_rng()

    h, w = template.shape
    x = rng.integers(50, IMAGE_SIZE[0] - 50 - w)
    y = rng.integers(50, IMAGE_SIZE[1] - 50 - h)

    return [x, y, x+w, y+h, templateName[1:-4]]

def CropTemplate(template):
    # Check if there are any completely white rows in the template images and 
    # remove them.
    top_row = 0
    bottom_row = -1
    left_column = 0
    right_column = -1

    while (True):
        if np.any(template[top_row] < 200):
            break
        else:
            top_row += 1
    while (True):
        if np.any(template[bottom_row] < 200):
            break
        else:
            bottom_row -= 1
    while (True):
        if np.any(template[:,left_column] < 200):
            break
        else:
            left_column += 1
    while (True):
        if np.any(template[:,right_column] < 200):
            break
        else:
            right_column -= 1

    if bottom_row == -1 and right_column == -1:
        out_image = template[top_row:, left_column:]
    elif bottom_row == -1:
        out_image = template[top_row:, left_column:right_column]
    elif right_column == -1:
        out_image = template[top_row:bottom_row, left_column:]
    else:
        bottom_row += 1
        right_column += 1
        out_image = template[top_row:bottom_row, left_column:right_column]
    return out_image

def GenerateOutput(name, image, boundingBoxes):
    
    if TRAIN:
        cv2.imwrite(os.path.join(TRAIN_IMAGES_FOLDER, name + ".png"), image)
        file = open(os.path.join(TRAIN_LABELS_FOLDER, name + ".txt"), "w")
    else:
        cv2.imwrite(os.path.join(VAL_IMAGES_FOLDER, name + ".png"), image)
        file = open(os.path.join(VAL_LABELS_FOLDER, name + ".txt"), "w")
    

    def GenerateLabel(bb):
        x = ((bb[2] + bb[0]) / 2) / IMAGE_SIZE[0]
        y = ((bb[3] + bb[1]) / 2) / IMAGE_SIZE[1]
        w = (bb[2] - bb[0]) / IMAGE_SIZE[0]
        h = (bb[3] - bb[1]) / IMAGE_SIZE[1]

        return [str(LABELS.index(bb[4])), str(x), str(y), str(w), str(h)]

    for bb in boundingBoxes:
        label = " ".join(GenerateLabel(bb))
        file.write(label + "\n")

    file.close()

def apply_gray_areas(image, template, x1, x2, y1, y2):
    # Insert the gray areas of the template to the background image thus 
    # preventing white background being inserted over other parts
    temp = np.copy(image[y1:y2, x1:x2])
    ret, mask = cv2.threshold(template, 225, 255, cv2.THRESH_BINARY)
    mask = mask / 255
    mask = mask.astype('uint8')
    ret, part = cv2.threshold(template, 225, 255, cv2.THRESH_TOZERO_INV)
    temp = np.multiply(temp, mask)
    temp = np.add(temp, part)
    image[y1:y2, x1:x2] = temp
    return image

def apply_color_change(image):
    # Apply random color shift to the template
    color_change = np.random.randint(-10, 11)
    if color_change < 0:
        c = np.zeros(image.shape, dtype=np.uint8) + -1*color_change
        np.putmask(image, c > image, c)
        image = image + color_change
    elif color_change > 0:
        c = np.ones(image.shape, dtype=np.uint8) * 255 - color_change
        np.putmask(image, c < image, c)
        image = image + color_change
    return image


def generate_images(number_of_images, offset):
    templates = GetTemplates()
    template_image = cv2.resize(cv2.cvtColor(cv2.imread(TEMPLATE_IMAGE), cv2.COLOR_BGR2GRAY), (IMAGE_SIZE))

    label_counts = {}
    for label in templates:
        label_counts[label] = 0

    for i in range(number_of_images):
        rng = np.random.default_rng()
        image = np.copy(template_image)

        # Get the least occuring parts
        temp = sorted(set(label_counts.values()))[:K]
        low_labels = [key for key in label_counts if label_counts[key] in temp]

        # Determine the amount of parts in this image
        objects = rng.integers(2, MAX_NUMBER_OF_OBJECTS, endpoint=True)
        boundingBoxes = []

        trials = 0
        while len(boundingBoxes) < objects:
            # Try to place the object 100 times
            if trials > 100:
                templateName = rng.choice(templates)
            else:
                templateName = rng.choice(low_labels)

            template = GetTemplateImage(templateName)

            rotation = rng.integers(0, 360)
            template = ndimage.rotate(template, rotation, cval=255)
            template = CropTemplate(template)

            bb = GetBoundingBox(template, templateName)

            if BoundingBoxesOverlap(bb, boundingBoxes, image, template):
                trials += 1
                continue

            boundingBoxes.append(bb)

            template = apply_color_change(template)

            # Apply the templates gray areas to the output image
            image = apply_gray_areas(image, template, bb[0], bb[2], bb[1], bb[3]) 
            trials = 0
            label_counts[templateName] += 1


        image = cv2.resize(image, OUT_IMAGE_SIZE)
        GenerateOutput(str(i+offset), image, boundingBoxes)

if __name__ == '__main__':
    templates = GetTemplates()

    template_image = cv2.cvtColor(cv2.imread(TEMPLATE_IMAGE), cv2.COLOR_BGR2GRAY)

    num_images_per_process = int(NUMBER_OF_IMAGES / NUMBER_OF_PROCESSES)

    processes = []

    for i in range(NUMBER_OF_PROCESSES):
        p = mp.Process(target=generate_images, args=(num_images_per_process, i*num_images_per_process))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()
