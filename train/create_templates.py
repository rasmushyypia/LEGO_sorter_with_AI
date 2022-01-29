"""
LEGO part template creator script
@author: Samuli Pohjola & Eetu Manninen
"""

import cv2
import os
import sys

FILE_FOLDER = os.path.dirname(__file__)
INPUT_FOLDER = os.path.join(FILE_FOLDER, "orig_images")
LABEL_FOLDER = os.path.join(FILE_FOLDER, "obj_train_data")
OUTPUT_FOLDER = os.path.join(FILE_FOLDER, "templates")

VISUALIZE = False

def GetFileNames():
    files = os.listdir(INPUT_FOLDER)

    fileList = []

    for file in files:
        if file.endswith(".png"):
            fileList.append(file)

    return fileList


if __name__ == '__main__':
    files = GetFileNames()

    for file in files:

        image = cv2.imread(os.path.join(INPUT_FOLDER, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        labels = open(os.path.join(LABEL_FOLDER, file[:-4] + ".txt"))

        height, width  = image.shape

        for label in labels.readlines():
            if len(label) != 0:
                c, x, y, w, h = label.split()

                x = int(float(x) * width)
                y = int(float(y) * height)
                w = int(float(w) * width / 2)
                h = int(float(h) * height / 2)

                x1 = int(x - w)
                y1 = int(y - h)
                x2 = int(x + w)
                y2 = int(y + h)

                template = image[y1:y2, x1:x2]

                if VISUALIZE:
                    cv2.imshow("template", template)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        sys.exit(1)

                cv2.imwrite(os.path.join(OUTPUT_FOLDER, file), template)




