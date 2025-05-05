import os
import json
import cv2

class CertainGrid:
    def __init__(self, config_file):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(self.base_dir, config_file), "r") as file:
            config = json.load(file)

        self.image_paths = [os.path.join(self.base_dir, path) for path in config["image_paths"]]
        self.image_set = [cv2.imread(path) for path in self.image_paths]

        self.corner_points = {
            "TOP_LEFT": (342, 18),
            "TOP_RIGHT": (1072, 33),
            "BOTTOM_LEFT": (362, 691),
            "BOTTOM_RIGHT": (1009, 712),
        }

    def get_corner_points(self):
        return self.corner_points

    def get_images(self):
        return self.image_set





