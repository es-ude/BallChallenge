import logging
from pathlib import Path

import numpy as np


class Projector:
    def __init__(self, homography_matrix= np.diag([1.0, 1.0, 1.0])):
        self.homography_matrix: np.ndarray = homography_matrix
        self._csv_delimiter = ','

    def save_homography_matrix(self, path: Path):
        with open(path, "wb") as f:
            np.savetxt(f, self.homography_matrix, delimiter=self._csv_delimiter)
            logging.info(f"Saved homography matrix to {path}")

    def load_homography_matrix(self, path: Path):
        with open(path, "rb") as f:
            self.homography_matrix = np.loadtxt(f, delimiter=self._csv_delimiter)
            logging.info(f"Loaded homography matrix from {path}")
            logging.info(f"Homography matrix= {self.homography_matrix}")

    def empty_window(self):
        raise NotImplementedError()

    def show_dot_in_projector_view(self, proj_pt: tuple[int, int], radius = -1)-> None:
        raise NotImplementedError()

    def show_empty_view(self):
        raise NotImplementedError()

    def show_dot_in_camera_view(self, cam_pt: tuple[int, int])->None:
        raise NotImplementedError()

if __name__ == "__main__":
    ...


