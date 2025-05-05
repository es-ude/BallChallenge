import cv2
import os
import json
import pytest
import imageProcessing as ip
import CreateArtificialGrid as cag
from imageProcessing import NoBoxDetectedError
from CreateArtificialGrid import BoxOutOfRangeError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "config.json"), "r") as file:
    config = json.load(file)

IMAGE_PATHS = [os.path.join(BASE_DIR, path) for path in config["image_paths"]]
EMPTY_IMAGE_PATH = os.path.join(BASE_DIR, config["empty_image"])
EDGE_IMAGE_PATH = os.path.join(BASE_DIR, config["edge_image"])

IMAGE_SET = [cv2.imread(path) for path in IMAGE_PATHS]

TOP_LEFT = [(342, 18)]

@pytest.mark.parametrize("image_index, expected_bounds_in_cm", [
    (0, [(127, 15, 147, 35), (140, 15, 160, 35), (127, 32, 147, 52), (140, 32, 160, 52)]),
    (1, [(38, 12, 58, 32), (53, 12, 73, 32), (38, 22, 58, 42), (53, 22, 73, 42)]),
    (2, [(143, 116, 163, 136), (160, 116, 180, 136), (143, 126, 163, 146), (160, 126, 180, 146)])
])

def test_bounding_box_coordinates_cm(image_index, expected_bounds_in_cm):
    coordinates = cag.calculate_position_of_bounding_box(IMAGE_SET[image_index], TOP_LEFT)
    for coord, (x_min, y_min, x_max, y_max) in zip(coordinates, expected_bounds_in_cm):
        assert x_min <= coord[0] <= x_max, f"Failed for coordinates: {coord}, expected x in range ({x_min}, {x_max}), y in range ({y_min}, {y_max})"
        assert y_min <= coord[1] <= y_max, f"Failed for coordinates: {coord}, expected x in range ({x_min}, {x_max}), y in range ({y_min}, {y_max})"


def test_no_box_detected():
    picture_without_sandsack = cv2.imread(EMPTY_IMAGE_PATH)
    with pytest.raises(NoBoxDetectedError):
        cag.calculate_position_of_bounding_box(picture_without_sandsack, TOP_LEFT)

def test_box_out_of_range():
    picture_with_sandsack_on_the_edge = cv2.imread(EDGE_IMAGE_PATH)
    with pytest.raises(BoxOutOfRangeError):
        cag.calculate_position_of_bounding_box(picture_with_sandsack_on_the_edge, TOP_LEFT)
