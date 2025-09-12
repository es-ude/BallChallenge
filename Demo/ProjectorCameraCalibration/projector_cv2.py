import logging
from pathlib import Path

import cv2
import numpy as np

from Demo.ProjectorCameraCalibration.projector import Projector

class ProjectorCV2(Projector):
    def __init__(self, resolution = (1280, 800), homography_matrix = np.diag([1.0, 1.0, 1.0])):
        super().__init__(homography_matrix)
        self.resolution = resolution
        self.dot_radius = 20
        self.window_name = 'Projector'
        self.empty_window()




    def empty_window(self):
        logging.info(f"Create empty window")
        proj_img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.window_name, 1800, 0)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.window_name, proj_img)

    def show_dot_in_projector_view(self, proj_pt: tuple[int, int], radius = -1)-> None:
        if radius == -1:
            radius = self.dot_radius
        logging.debug(f"Project Point at {proj_pt} with Dot Radius: {radius}")
        proj_img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        cv2.circle(proj_img, proj_pt, radius, (0, 0, 255), -1)
        cv2.imshow(self.window_name, proj_img)


    def show_empty_view(self):
        logging.info(f"Show Empty Window")
        proj_img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        cv2.imshow(self.window_name, proj_img)

    def show_dot_in_camera_view(self, cam_pt: tuple[int, int])->None:
        cam_point = np.array([cam_pt[0], cam_pt[1], 1.0])
        proj_point_hom = self.homography_matrix @ cam_point
        proj_x :int = proj_point_hom[0].astype(int)
        proj_y : int = proj_point_hom[1].astype(int)
        logging.info(f"Show Dot In Camera View at camera position {cam_pt} and projection point {proj_x, proj_y}")
        self.show_dot((proj_x, proj_y))


if __name__ == '__main__':
    ...