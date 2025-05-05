import cv2
import os
import json
import numpy as np
import image_processing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "config.json"), "r") as file:
    config = json.load(file)

class BoxOutOfRangeError(Exception):
    def __init__(self, message="The box is not within the frame of the grid"):
         super().__init__(message)

ORIGINAL_IMAGE_PATH = os.path.join(BASE_DIR, config["original_image"])

TOP_LEFT = [(342, 18)]
TOP_RIGHT = [(1072, 33)]
BOTTOM_RIGHT = [(1009, 712)]
BOTTOM_LEFT = [(362, 691)]
PIXEL_PER_CM = 3.6
GRID_SIZE = 40

img = cv2.imread(ORIGINAL_IMAGE_PATH)

top_edge = np.linspace(TOP_LEFT, TOP_RIGHT, GRID_SIZE)
bottom_edge = np.linspace(BOTTOM_LEFT, BOTTOM_RIGHT, GRID_SIZE)

grid_points = []
for i in range(GRID_SIZE):
    column_points = np.linspace(top_edge[i], bottom_edge[i], GRID_SIZE)
    grid_points.append(column_points)

grid_points = np.array(grid_points)

for row in grid_points:
    for point in row:
        extract = point[0]
        x, y = int(extract[0]), int(extract[1])
        cv2.circle(img, (x, y), 2, (0, 0, 255), 2)

local_coordinate_system = grid_points - (342, 18)


def calculate_position_of_bounding_box(img, origin):
    x, y, w, h = image_processing.process_image(img)

    bounding_box_coordinates = [
    (x, y),
    (x + w, y),
    (x, y + h),
    (x + w, y + h),
    ]

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('GreyImage', img)
    cv2.waitKey(0)
    bounding_box_coordinates_cm = [((x - (origin[0])[0])/ PIXEL_PER_CM, (y - (origin[0])[1])/ PIXEL_PER_CM) for x, y in bounding_box_coordinates]

    for i in bounding_box_coordinates_cm:
        if i[0] < 0 or i[1] < 0:
            raise BoxOutOfRangeError()
        print(i)

    return bounding_box_coordinates_cm





