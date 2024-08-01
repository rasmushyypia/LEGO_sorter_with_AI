import cv2
import os

def read_info_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        roi_line = lines[0].strip()
        exposure_line = lines[1].strip()

        # Extracting ROI coordinates
        roi_str = roi_line.split(': ')[1].strip('()')
        roi = tuple(map(int, roi_str.split(', ')))

        # Extracting Exposure Time
        exposure_time = int(exposure_line.split(': ')[1])

    return roi, exposure_time

def crop_image(image, roi):
    # Get ROI coordinates as (x1, y1, x2, y2)
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1

    # Crop the image using ROI
    cropped_image = image[y1:y1+h, x1:x1+w]

    return cropped_image

def process_images_in_folder(source_folder, dest_folder, roi):
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(source_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load image {image_path}. Skipping.")
                continue

            cropped_image = crop_image(image, roi)

            # Save the cropped image to the destination folder
            cropped_image_path = os.path.join(dest_folder, filename)
            cv2.imwrite(cropped_image_path, cropped_image)
            print(f"Cropped image saved as {cropped_image_path}")

def main():
    # File paths
    info_file_path = 'src/data/background_image_0_info.txt'
    source_folder = 'src/data/testikuvat/original'
    dest_folder = 'src/data/testikuvat/cropped'

    # Read info from file
    roi, exposure_time = read_info_from_file(info_file_path)
    print(f"ROI: {roi}, Exposure Time: {exposure_time}")

    # Process images in the source folder
    process_images_in_folder(source_folder, dest_folder, roi)

if __name__ == "__main__":
    main()
