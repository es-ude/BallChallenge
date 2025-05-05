import pytest
import os
import json
import cv2
import divideGrid
from certainGridConfig import CertainGrid

config = CertainGrid("config.json")
corner_points_list = list(config.get_corner_points().values())


def test_intersection():
    calculated_value = divideGrid.intersection(6, 4, 8, 8, 0, 0, 10, 10)
    assert calculated_value <= 8

def test_divide_grid():
        calculated_label_top_right = divideGrid.divide_grid(corner_points_list, config.get_images()[0])
        assert calculated_label_top_right <= "TopRight"
        calculated_label_top_left = divideGrid.divide_grid(corner_points_list, config.get_images()[1])
        assert calculated_label_top_left <= "TopLeft"
        calculated_label_bottom_right = divideGrid.divide_grid(corner_points_list, config.get_images()[2])
        assert calculated_label_bottom_right <= "BottomRight"

