import numpy as np
import cv2
import os
from scipy import ndimage
import multiprocessing as mp
import time
import yaml

# Constants
IMAGE_SIZE = (1920, 1920)
OUT_IMAGE_SIZE = (640, 640)
MAX_NUMBER_OF_OBJECTS = 10
TRAINING_IMAGES = 14000
VALIDATION_IMAGES = 4000
NUMBER_OF_PROCESSES = 8
K = 3

# Directories
FILE_FOLDER = os.path.dirname(__file__)
TEMPLATE_FOLDER = os.path.join(FILE_FOLDER, "data", "templates")
TRAIN_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "images", "train")
TRAIN_LABELS_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "labels", "train")
VAL_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "images", "val")
VAL_LABELS_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "labels", "val")
TEMPLATE_IMAGE = os.path.join(FILE_FOLDER, "data", "background.png")

# Path to the data.yaml file
DATA_YAML_PATH = os.path.join(FILE_FOLDER, "data.yaml")

def read_labels_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

# Read labels from data.yaml
LABELS = read_labels_from_yaml(DATA_YAML_PATH)

def create_directories():
    os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_LABELS_FOLDER, exist_ok=True)
    os.makedirs(VAL_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(VAL_LABELS_FOLDER, exist_ok=True)

def get_templates(template_folder):
    if not os.path.exists(template_folder):
        raise ValueError(f"Template folder '{template_folder}' does not exist.")
        
    files = os.listdir(template_folder)
    file_list = [file for file in files if file.endswith(".jpg")]
    
    if not file_list:
        raise ValueError(f"No JPG files found in the template folder '{template_folder}'.")
    
    return file_list

def get_template_image(template_folder, name):
    image = cv2.imread(os.path.join(template_folder, name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def check_overlapping_gray_areas(image, template, bb):
    _, mask = cv2.threshold(template, 225, 255, cv2.THRESH_BINARY)
    mask = mask / 255
    mask = mask.astype('uint8')
    _, mask2 = cv2.threshold(image[bb[1]:bb[3], bb[0]:bb[2]], 225, 255, cv2.THRESH_BINARY)
    mask2 = mask2 / 255
    mask2 = mask2.astype('uint8')
    overlaps = np.sum(np.multiply(mask, mask2)) > 3
    return overlaps

def check_overlapping_bounding_boxes(bb1, bb2):
    return not ((bb1[0] >= bb2[2]) or (bb1[2] <= bb2[0]) or (bb1[3] <= bb2[1]) or (bb1[1] >= bb2[3]))

def bounding_boxes_overlap(bb, bounding_boxes, image, template):
    if len(bounding_boxes) > 0:
        for b in bounding_boxes:
            if check_overlapping_bounding_boxes(bb, b):
                if check_overlapping_gray_areas(image, template, bb):
                    return True
    return False

def get_bounding_box(template, template_name):
    rng = np.random.default_rng()
    h, w = template.shape
    x = rng.integers(50, IMAGE_SIZE[0] - 50 - w)
    y = rng.integers(50, IMAGE_SIZE[1] - 50 - h)
    
    # Extract the numeric part after 'lego_'
    label_part = template_name.split('-')[1]
    label = label_part.split('_')[1]
    
    if not label.isnumeric():
        raise ValueError(f"Label {label} is not a valid numeric label.")
    
    return [x, y, x + w, y + h, int(label)]

def crop_template(template):
    top_row = 0
    bottom_row = template.shape[0] - 1
    left_column = 0
    right_column = template.shape[1] - 1

    while top_row < template.shape[0] and np.all(template[top_row] >= 200):
        top_row += 1
    while bottom_row >= 0 and np.all(template[bottom_row] >= 200):
        bottom_row -= 1
    while left_column < template.shape[1] and np.all(template[:, left_column] >= 200):
        left_column += 1
    while right_column >= 0 and np.all(template[:, right_column] >= 200):
        right_column -= 1

    if top_row > bottom_row or left_column > right_column:
        return template

    return template[top_row:bottom_row + 1, left_column:right_column + 1]

def generate_output(name, image, bounding_boxes, output_folder, label_folder):
    cv2.imwrite(os.path.join(output_folder, name + ".png"), image)
    with open(os.path.join(label_folder, name + ".txt"), "w") as file:
        def generate_label(bb):
            x = ((bb[2] + bb[0]) / 2) / IMAGE_SIZE[0]
            y = ((bb[3] + bb[1]) / 2) / IMAGE_SIZE[1]
            w = (bb[2] - bb[0]) / IMAGE_SIZE[0]
            h = (bb[3] - bb[1]) / IMAGE_SIZE[1]
            return [str(bb[4]), str(x), str(y), str(w), str(h)]

        for bb in bounding_boxes:
            label = " ".join(generate_label(bb))
            file.write(label + "\n")

def apply_gray_areas(image, template, x1, x2, y1, y2):
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
    color_change = np.random.randint(-10, 11)
    if color_change < 0:
        c = np.zeros(image.shape, dtype=np.uint8) + -1 * color_change
        np.putmask(image, c > image, c)
        image = image + color_change
    elif color_change > 0:
        c = np.ones(image.shape, dtype=np.uint8) * 255 - color_change
        np.putmask(image, c < image, c)
        image = image + color_change
    return image

def generate_images(number_of_images, offset, template_folder, background_image_path, output_folder, label_folder, counter):
    templates = get_templates(template_folder)
    
    if not templates:
        raise ValueError("No templates found in the template folder. Ensure the template folder is populated with template images.")

    template_image = cv2.resize(cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2GRAY), IMAGE_SIZE)

    label_counts = {label: 0 for label in templates}

    for i in range(number_of_images):
        rng = np.random.default_rng()
        image = np.copy(template_image)
        temp = sorted(set(label_counts.values()))[:K]
        low_labels = [key for key in label_counts if label_counts[key] in temp]
        objects = rng.integers(2, MAX_NUMBER_OF_OBJECTS, endpoint=True)
        bounding_boxes = []
        trials = 0

        while len(bounding_boxes) < objects:
            if trials > 100 or not low_labels:
                template_name = rng.choice(templates)
            else:
                template_name = rng.choice(low_labels)

            template = get_template_image(template_folder, template_name)
            rotation = rng.integers(0, 360)
            template = ndimage.rotate(template, rotation, cval=255)
            template = crop_template(template)
            bb = get_bounding_box(template, template_name)

            if bounding_boxes_overlap(bb, bounding_boxes, image, template):
                trials += 1
                continue

            bounding_boxes.append(bb)
            template = apply_color_change(template)
            image = apply_gray_areas(image, template, bb[0], bb[2], bb[1], bb[3])
            trials = 0
            label_counts[template_name] += 1

        image = cv2.resize(image, OUT_IMAGE_SIZE)
        generate_output(str(i + offset), image, bounding_boxes, output_folder, label_folder)
        with counter.get_lock():
            counter.value += 1

def display_progress(counter, total_images, phase):
    while counter.value < total_images:
        print(f'Progress: {counter.value}/{total_images} {phase} images generated', end='\r', flush=True)
        time.sleep(1)
    print(f'Progress: {counter.value}/{total_images} {phase} images generated', flush=True)

if __name__ == '__main__':
    create_directories()

    # Counters for progress tracking
    train_counter = mp.Value('i', 0)
    val_counter = mp.Value('i', 0)

    # Generate training images
    num_training_images_per_process = int(TRAINING_IMAGES / NUMBER_OF_PROCESSES)
    total_train_images = TRAINING_IMAGES
    processes = []

    # Start a separate process for displaying progress of training images
    train_progress_process = mp.Process(target=display_progress, args=(train_counter, total_train_images, "training"))
    train_progress_process.start()

    for i in range(NUMBER_OF_PROCESSES):
        p = mp.Process(target=generate_images, args=(
            num_training_images_per_process, i * num_training_images_per_process, TEMPLATE_FOLDER, TEMPLATE_IMAGE,
            TRAIN_IMAGES_FOLDER, TRAIN_LABELS_FOLDER, train_counter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    train_progress_process.join()

    # Generate validation images
    num_validation_images_per_process = int(VALIDATION_IMAGES / NUMBER_OF_PROCESSES)
    total_val_images = VALIDATION_IMAGES
    processes = []

    # Start a separate process for displaying progress of validation images
    val_progress_process = mp.Process(target=display_progress, args=(val_counter, total_val_images, "validation"))
    val_progress_process.start()

    for i in range(NUMBER_OF_PROCESSES):
        p = mp.Process(target=generate_images, args=(
            num_validation_images_per_process, i * num_validation_images_per_process + TRAINING_IMAGES, TEMPLATE_FOLDER,
            TEMPLATE_IMAGE, VAL_IMAGES_FOLDER, VAL_LABELS_FOLDER, val_counter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    val_progress_process.join()
