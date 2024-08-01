import os
import yaml

FILE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(FILE_FOLDER, "data")
IMAGE_FILES_LOCATION = os.path.join(DATA_FOLDER, "orig_images")
OUTPUT_FILE = os.path.join(IMAGE_FILES_LOCATION, "image_filenames.txt")
DATA_YAML_PATH = os.path.join(DATA_FOLDER, "data.yaml")

def main():
    print(f"FILE_FOLDER: {FILE_FOLDER}")
    print(f"DATA_FOLDER: {DATA_FOLDER}")
    print(f"IMAGE_FILES_LOCATION: {IMAGE_FILES_LOCATION}")
    print(f"OUTPUT_FILE: {OUTPUT_FILE}")

    if not os.path.exists(IMAGE_FILES_LOCATION):
        print(f"Error: {IMAGE_FILES_LOCATION} does not exist.")
        return

    if not os.path.isdir(IMAGE_FILES_LOCATION):
        print(f"Error: {IMAGE_FILES_LOCATION} is not a directory.")
        return

    image_files = [f for f in os.listdir(IMAGE_FILES_LOCATION) 
                   if os.path.isfile(os.path.join(IMAGE_FILES_LOCATION, f)) 
                   and (f.lower().endswith('.jpg') or f.lower().endswith('.png'))]

    labels = []
    with open(OUTPUT_FILE, 'w') as file:
        for image_file in image_files:
            label = os.path.splitext(image_file)[0]  # Strip the extension
            file.write(f"{label}\n")
            labels.append(label)
            print(label)  # Optional: print the label to the console

    create_data_yaml(labels)

def create_data_yaml(labels):
    data = {
        'train': os.path.join(DATA_FOLDER, 'dataset', 'images', 'train'),
        'val': os.path.join(DATA_FOLDER, 'dataset', 'images', 'val'),
        'nc': len(labels),
        'names': labels
    }

    with open(DATA_YAML_PATH, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    print(f"data.yaml created at: {DATA_YAML_PATH}")

if __name__ == "__main__":
    main()
