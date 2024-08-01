import cv2
import os

FILE_FOLDER = os.path.dirname(__file__)
INPUT_FOLDER = os.path.join(FILE_FOLDER, "data", "annotated_data", "images")
LABEL_FOLDER = os.path.join(FILE_FOLDER, "data", "annotated_data", "labels")
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "data", "templates")

def check_folders():
    """Check if input and label folders exist."""
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"Input folder '{INPUT_FOLDER}' does not exist")
    if not os.path.exists(LABEL_FOLDER):
        raise FileNotFoundError(f"Label folder '{LABEL_FOLDER}' does not exist")

def get_file_names():
    """Retrieve all .jpg filenames from the INPUT_FOLDER."""
    files = os.listdir(INPUT_FOLDER)
    return [file for file in files if file.endswith(".jpg")]

def simplify_name(filename):
    """Simplify the filename by removing characters up to the last dash and removing the last two number groups."""
    # Remove characters up to the last dash
    return filename.split('-')[-1]

def create_templates():
    files = get_file_names()

    if not files:
        print(f"No .jpg files found in '{INPUT_FOLDER}'")
        return

    for file in files:
        try:
            image_path = os.path.join(INPUT_FOLDER, file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image {file}")
                continue

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            label_path = os.path.join(LABEL_FOLDER, file[:-4] + ".txt")
            if not os.path.exists(label_path):
                print(f"Label file '{label_path}' does not exist")
                continue

            with open(label_path, 'r') as labels:
                height, width = image_gray.shape

                for label in labels:
                    if len(label.strip()) == 0:
                        continue

                    c, x, y, w, h = map(float, label.split())
                    x = int(x * width)
                    y = int(y * height)
                    w = int(w * width / 2)
                    h = int(h * height / 2)

                    x1 = max(0, x - w)
                    y1 = max(0, y - h)
                    x2 = min(width, x + w)
                    y2 = min(height, y + h)

                    template = image_gray[y1:y2, x1:x2]

                    # Simplify the filename
                    simplified_name = simplify_name(file)
                
                    # Ensure the output folder exists
                    if not os.path.exists(OUTPUT_FOLDER):
                        os.makedirs(OUTPUT_FOLDER)

                    template_path = os.path.join(OUTPUT_FOLDER, simplified_name)
                    cv2.imwrite(template_path, template)
                    print(f"Template saved: {simplified_name}")

        except Exception as e:
            print(f"An error occurred processing file {file}: {e}")

def main():
    try:
        check_folders()
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        create_templates()
    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
