import numpy as np
import cv2
import os
from scipy import ndimage
import multiprocessing as mp
import time

# Constants
IMAGE_SIZE = (1920, 1920)
OUT_IMAGE_SIZE = (640, 640)
MAX_NUMBER_OF_OBJECTS = 10
TRAINING_IMAGES = 12000
VALIDATION_IMAGES = 4000
NUMBER_OF_PROCESSES = 8
K = 3
MAX_TRIALS = 500  # Maximum number of trials for placing an object

# Margin Constants
MARGIN_X = 10
MARGIN_Y = 50

# Directories
FILE_FOLDER = os.path.dirname(__file__)
TEMPLATE_FOLDER = os.path.join(FILE_FOLDER, "data", "templates")
TRAIN_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "images", "train")
TRAIN_LABELS_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "labels", "train")
VAL_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "images", "val")
VAL_LABELS_FOLDER = os.path.join(FILE_FOLDER, "data", "dataset", "labels", "val")
TEMPLATE_IMAGE = os.path.join(FILE_FOLDER, "data", "background_image_0.png")

# Path to the classes.txt file
CLASSES_TXT_PATH = os.path.join(FILE_FOLDER, "data", "annotated_data", "classes.txt")

def read_labels_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

# Read labels from classes.txt
LABELS = read_labels_from_txt(CLASSES_TXT_PATH)

def create_directories():
    os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_LABELS_FOLDER, exist_ok=True)
    os.makedirs(VAL_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(VAL_LABELS_FOLDER, exist_ok=True)

def get_templates(template_folder):
    """
    Retrieves all template filenames from the specified template folder.
    Raises a ValueError if the folder does not exist or contains no JPG files.
    """
    if not os.path.exists(template_folder):
        raise ValueError(f"Template folder '{template_folder}' does not exist.")
        
    files = os.listdir(template_folder)
    file_list = [file for file in files if file.endswith(".jpg")]
    
    if not file_list:
        raise ValueError(f"No JPG files found in the template folder '{template_folder}'.")
    
    return file_list

def get_template_image(template_folder, name):
    """
    Reads a template image from the specified folder and converts it to grayscale.
    Returns the grayscale image.
    """
    image = cv2.imread(os.path.join(template_folder, name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def check_overlapping_gray_areas(image, template, bb):
    """
    Checks if the grayscale areas of the template overlap with the specified bounding box region in the image.
    Converts both the template and the region of the image to binary masks, then checks for overlap.
    
    Args:
        image (numpy.ndarray): The background image where the template is being placed.
        template (numpy.ndarray): The template image being placed.
        bb (list): The bounding box [x1, y1, x2, y2] defining the region in the image to check for overlap.
        
    Returns:
        bool: True if there is a significant overlap of grayscale areas, otherwise False.
    """
    # Convert the template to a binary mask where white areas are set to 1 and other areas are set to 0
    _, mask = cv2.threshold(template, 225, 255, cv2.THRESH_BINARY)
    mask = mask / 255
    mask = mask.astype('uint8')

    # Convert the specified region of the background image to a binary mask
    _, mask2 = cv2.threshold(image[bb[1]:bb[3], bb[0]:bb[2]], 225, 255, cv2.THRESH_BINARY)
    mask2 = mask2 / 255
    mask2 = mask2.astype('uint8')

    # Multiply the two masks to find overlapping areas
    overlaps = np.sum(np.multiply(mask, mask2)) > 3
    
    return overlaps

def check_overlapping_bounding_boxes(bb1, bb2):
    """
    Checks if two bounding boxes overlap.
    
    Args:
        bb1 (list): The first bounding box [x1, y1, x2, y2].
        bb2 (list): The second bounding box [x1, y1, x2, y2].
    
    Returns:
        bool: True if the bounding boxes overlap, otherwise False.
    """
    return not ((bb1[0] >= bb2[2]) or (bb1[2] <= bb2[0]) or (bb1[3] <= bb2[1]) or (bb1[1] >= bb2[3]))

def bounding_boxes_overlap(bb, bounding_boxes, image, template):
    """
    Determines if the given bounding box overlaps with any bounding boxes in the list using both bounding box and gray area checks.
    
    Args:
        bb (list): The bounding box to check [x1, y1, x2, y2].
        bounding_boxes (list): The list of existing bounding boxes.
        image (numpy.ndarray): The background image where the template is being placed.
        template (numpy.ndarray): The template image being placed.
    
    Returns:
        bool: True if there is significant overlap, otherwise False.
    """
    if len(bounding_boxes) > 0:
        for b in bounding_boxes:
            if check_overlapping_bounding_boxes(bb, b):
                if check_overlapping_gray_areas(image, template, bb):
                    return True
    return False

def get_bounding_box(template, template_name):
    """
    Generates a random bounding box for a given template image, ensuring it is placed randomly within the defined image size and extracts the label from the template name.
    
    Args:
        template (numpy.ndarray): The template image.
        template_name (str): The name of the template image file.
    
    Returns:
        list: The bounding box [x1, y1, x2, y2, label_index].
    """
    rng = np.random.default_rng()
    h, w = template.shape
    x = rng.integers(MARGIN_X, IMAGE_SIZE[0] - MARGIN_X - w)
    y = rng.integers(MARGIN_Y, IMAGE_SIZE[1] - MARGIN_Y - h)
    
    # Extract the label from the template name
    label = template_name.split('.')[0]
    
    if label not in LABELS:
        raise ValueError(f"Label {label} is not found in the provided classes.txt.")
    
    return [x, y, x + w, y + h, LABELS.index(label)]


def crop_template(template, border_value=200):
    """
    Crops the template image to remove any border areas with pixel values greater than or equal to the specified border_value.
    This helps in focusing on the actual object in the template and removing unnecessary white space.

    Args:
        template (numpy.ndarray): The template image to be cropped.
        border_value (int): The pixel value threshold for determining the border. Default is 200.

    Returns:
        numpy.ndarray: The cropped template image.
    """
    top_row = 0
    bottom_row = template.shape[0] - 1
    left_column = 0
    right_column = template.shape[1] - 1

    # Increment top_row until a row with pixel values less than border_value is found
    while top_row < template.shape[0] and np.all(template[top_row] >= border_value):
        top_row += 1

    # Decrement bottom_row until a row with pixel values less than border_value is found
    while bottom_row >= 0 and np.all(template[bottom_row] >= border_value):
        bottom_row -= 1

    # Increment left_column until a column with pixel values less than border_value is found
    while left_column < template.shape[1] and np.all(template[:, left_column] >= border_value):
        left_column += 1

    # Decrement right_column until a column with pixel values less than border_value is found
    while right_column >= 0 and np.all(template[:, right_column] >= border_value):
        right_column -= 1

    # If the cropped region is invalid, return the original template
    if top_row > bottom_row or left_column > right_column:
        return template

    # Return the cropped template
    return template[top_row:bottom_row + 1, left_column:right_column + 1]

def generate_output(name, image, bounding_boxes, output_folder, label_folder):
    """
    Saves the generated synthetic image and its corresponding bounding box annotations.
    
    Args:
        name (str): The name of the output file (without extension).
        image (numpy.ndarray): The synthetic image.
        bounding_boxes (list): The list of bounding boxes [x1, y1, x2, y2, label_index].
        output_folder (str): The folder to save the output image.
        label_folder (str): The folder to save the output label.
    """
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
    """
    Applies the template onto the background image, ensuring the template is blended correctly.
    
    Args:
        image (numpy.ndarray): The background image where the template is being placed.
        template (numpy.ndarray): The template image being placed.
        x1 (int): The x-coordinate of the top-left corner of the bounding box.
        x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
        y1 (int): The y-coordinate of the top-left corner of the bounding box.
        y2 (int): The y-coordinate of the bottom-right corner of the bounding box.
    
    Returns:
        numpy.ndarray: The image with the template applied.
    """
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
    """
    Applies a random change in intensity to the grayscale image to simulate different lighting conditions.
    The change is a random integer between -10 and 10:
    - Negative values darken the image.
    - Positive values brighten the image.
    The function ensures pixel values remain within the valid range (0 to 255).
    """
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
    """
    Generates synthetic images by placing randomly transformed templates onto a background image.
    Ensures that no significant overlap occurs between objects.

    Args:
        number_of_images (int): The number of images to generate.
        offset (int): The offset for naming the output files.
        template_folder (str): The folder containing the template images.
        background_image_path (str): The path to the background image.
        output_folder (str): The folder to save the generated images.
        label_folder (str): The folder to save the generated labels.
        counter (multiprocessing.Value): A counter for tracking progress.
    """
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
            if trials > MAX_TRIALS or not low_labels:
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
                if trials > MAX_TRIALS:
                    print(f"Max trials exceeded for image {i + offset}. Moving to the next image.")
                    break
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
    """
    Displays the progress of image generation.

    Args:
        counter (multiprocessing.Value): A counter for tracking progress.
        total_images (int): The total number of images to be generated.
        phase (str): The phase of image generation (e.g., "training" or "validation").
    """
    while counter.value < total_images:
        print(f'Progress: {counter.value}/{total_images} {phase} images generated', end='\r', flush=True)
        time.sleep(1)

if __name__ == '__main__':
    create_directories()

    # Counters for progress tracking
    train_counter = mp.Value('i', 0)
    val_counter = mp.Value('i', 0)

    # Generate training images
    num_training_images_per_process = TRAINING_IMAGES // NUMBER_OF_PROCESSES
    remaining_training_images = TRAINING_IMAGES % NUMBER_OF_PROCESSES
    total_train_images = TRAINING_IMAGES
    processes = []

    # Start a separate process for displaying progress of training images
    train_progress_process = mp.Process(target=display_progress, args=(train_counter, total_train_images, "training"))
    train_progress_process.start()

    for i in range(NUMBER_OF_PROCESSES):
        num_images = num_training_images_per_process + (1 if i < remaining_training_images else 0)
        p = mp.Process(target=generate_images, args=(
            num_images, sum(num_training_images_per_process + (1 if j < remaining_training_images else 0) for j in range(i)),
            TEMPLATE_FOLDER, TEMPLATE_IMAGE, TRAIN_IMAGES_FOLDER, TRAIN_LABELS_FOLDER, train_counter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    train_progress_process.join()

    # Generate validation images
    num_validation_images_per_process = VALIDATION_IMAGES // NUMBER_OF_PROCESSES
    remaining_validation_images = VALIDATION_IMAGES % NUMBER_OF_PROCESSES
    total_val_images = VALIDATION_IMAGES
    processes = []

    # Start a separate process for displaying progress of validation images
    val_progress_process = mp.Process(target=display_progress, args=(val_counter, total_val_images, "validation"))
    val_progress_process.start()

    for i in range(NUMBER_OF_PROCESSES):
        num_images = num_validation_images_per_process + (1 if i < remaining_validation_images else 0)
        p = mp.Process(target=generate_images, args=(
            num_images, TRAINING_IMAGES + sum(num_validation_images_per_process + (1 if j < remaining_validation_images else 0) for j in range(i)),
            TEMPLATE_FOLDER, TEMPLATE_IMAGE, VAL_IMAGES_FOLDER, VAL_LABELS_FOLDER, val_counter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    val_progress_process.join()
