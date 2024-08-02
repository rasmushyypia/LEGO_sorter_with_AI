# LEGO Detection and Sorting Pipeline

This repository provides a comprehensive pipeline for capturing images, labeling data, generating synthetic training data, and training a YOLO model to detect and sort LEGO pieces. Follow the steps outlined below to set up and use the pipeline effectively.

## Table of Contents
- [Installation](#installation)
- [Step 1: Capture Calibration Image](#step-1-capture-calibration-image)
- [Step 2: Calibrate the Camera](#step-2-calibrate-the-camera)
- [Step 3: Capture ROI (Region of Interest) Image](#step-3-capture-roi-region-of-interest-image)
- [Step 4: Create LEGO Dataset](#step-4-create-lego-dataset)
- [Step 5: Generate Label Names](#step-5-generate-label-names)
- [Step 6: Label the LEGO Pieces](#step-6-label-the-lego-pieces)
- [Step 7: Create Templates](#step-7-create-templates)
- [Step 8: Generate Synthetic Images](#step-8-generate-synthetic-images)
- [Step 9: Train the YOLO Model](#step-9-train-the-yolo-model)
- [Step 10: Use the Trained Model for Predictions](#step-10-use-the-trained-model-for-predictions)
- [Contributing](#contributing)

## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/lego-detection-pipeline.git
cd lego-detection-pipeline
pip install -r requirements.txt
```

## step 1: capture calibration image

run `camera_gui.py`:

- **calibration prompt:** when prompted about using the calibration file, answer "no."
- **place calibration board:** position the calibration board under the camera's field of view (fov).
- **adjust exposure:** use the gui to adjust the exposure slider until the calibration board is clearly visible.
- **capture calibration image:** click the "capture full image" button to save an image of the calibration board for calibration.
<img src="/media/calibration_image.png" alt="Calibration Image" width="500">


## step 2: calibrate the camera

run `camera_calibration.py`:

- **calibration processing:** the script processes the calibration image, calculates the camera matrix and distortion coefficients, and saves this data for later use.
- **note:** if the program outputs "no checkerboard found!", ensure the `chessboard_size` variable matches your calibration board and all corners are visible in the image.
<img src="/media/calibrated_image.png" alt="Calibration Image" width="500">

## step 3: capture roi (region of interest) image

run `camera_gui.py` again:

- **calibration data:** ensure `calibration_input_location` matches the calibration data `.npz` file created in the previous step.
- **calibration prompt:** when prompted, choose "yes" to use the calibration file.
- **adjust roi:**
  - adjust the exposure if necessary to get a clear view.
  - select the appropriate area under the backlight using the gui. example settings: `exposure: 120000; roi_size: 1550`.
  - click the "capture roi" button to save an image of the empty background along with roi and exposure information.
<img src="/media/roi_image.png" alt="Calibration Image" width="500">

## step 4: create lego dataset

**Lego part id management**:

`TAULegoPartIDs.xlsx` file contains the ID information for all lego parts. Ensure that each part ID is unique and consistent with this file.  
When adding new parts, assign a new ID following the existing naming scheme and update the `TAULegoPartIDs.xlsx` file.

run `lego_dataset_creator.py` to open the graphical user interface.

- click **set new ID (n)** to assign a new ID to the lego part.
- use the **select color** dropdown to choose the color of the lego part.

**capture standard orientations**:

- position the lego part and click **capture image (c)** to capture images from different angles.
- the image will be saved as `<part_id>_<color>_<orientation>.jpg` in the `data/orig_images` directory.

**capture unusual orientations**:

- for challenging angles, click **capture weird orientation (u)** to take images that might confuse the model with other parts.
  
  <img src="/media/data_generator_image.png" alt="Calibration Image" width="500">

## step 5: generate label names

run `generate_label_names.py`:

- **functionality:** this script gathers names of all the original images and writes them to a text file, `image_filenames.txt`.
- **purpose:** the generated text file can be used to add label names in label studio without needing to manually type them out.

## Step 6: Label the LEGO Pieces

### Install Label Studio:

- **Installation Guide:** Follow the Label Studio [installation guide](https://labelstud.io/guide/install.html) to set up Label Studio on your machine.

### Set Up Label Studio Project:

- **Project Creation:** Create a new project in Label Studio to manage your LEGO image labeling task.
- **Image Loading:** Configure the project to load images from the directory containing your LEGO images (`data/orig_images`).

### Define Labels:

- **Label Import:** Use the text file generated by `generate_label_names.py` to import label names directly into Label Studio.

### Label the Images:

- **Draw Bounding Boxes:** Open each image in Label Studio and draw a bounding box around each LEGO piece.
- **Assign Labels:** Assign the correct label to each bounding box.


## step 7: create templates

export the labeled images:

- **export:** export the labeled images in YOLO format from label-studio.
- **unzip:** place the zip file in the `data/annotated_data` folder and unzip it.

run `create_templates.py`:

- **template creation:** this script processes the annotated images and their corresponding labels to create template images based on the bounding boxes.
- **output:** the templates are saved in the `data/templates` directory.

## step 8: generate synthetic images

run `create_images.py`:

- **background and templates:** the script reads the background image and templates.
- **template placement:** it places multiple templates on the background image to create synthetic images.
- **non-overlapping:** ensures that the templates do not overlap significantly and stay within the margins.
- **augmentation:** applies random rotations and intensity changes to templates for augmentation.
- **image generation:** generates a specified number of training and validation images.
- **output:** saves the synthetic images and their corresponding labels in the appropriate directories.

## step 9: train the yolo model

### prepare for training:

- **ensure data.yaml is configured correctly:** verify that `data.yaml` contains the correct paths to your training and validation datasets.

### run training script:

use the following script to train your yolo model:

```python
from ultralytics import YOLO
import os

# load the pretrained yolov8 model
model = YOLO('yolov8n.pt')

# define the dataset path
current_path = os.getcwd()
dataset_path = os.path.join(current_path, 'src', 'data', 'data.yaml')

# train the model
results = model.train(data=dataset_path, epochs=50, imgsz=640, batch=32)
```
more info on the model training settings and hyperparameters on [ultralytics website](https://docs.ultralytics.com/modes/train/#train-settings)
