import os
import cv2
import csv
import divideGrid
from certainGridConfig import CertainGrid
from imageProcessing import NoBoxDetectedError

#the code only works if it is in a folder with another folder in it called "SensorValues"
if __name__ == "__main__":
    config = CertainGrid("config.json")
    corner_points_list = list(config.get_corner_points().values())

def is_directory_empty(directory_path):
    return len(os.listdir(directory_path)) == 0

def contains_image(path_to_folder):
    for file in os.listdir(path_to_folder):
        if file.lower().endswith(".jpeg"):
            return True
    return False

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    new_folder_path = os.path.join(base_dir, "SensorValues")

    for entry in os.scandir(new_folder_path):
        if entry.is_dir():
            if not is_directory_empty(entry.path):
                get_image_path = contains_image(entry.path)
                if get_image_path:
                    image = cv2.imread(os.path.join(entry.path, "image.jpeg"))
                    if image is not None:
                        try:
                            label = divideGrid.divide_grid(corner_points_list, image)
                            with open(os.path.join(entry.path, "measurement.csv"), mode="a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow([label])
                        except NoBoxDetectedError:
                            continue



