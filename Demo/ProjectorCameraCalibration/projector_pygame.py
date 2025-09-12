import logging
import time

import numpy as np
import pygame
import pygame.display

from Demo.ProjectorCameraCalibration.projector import Projector


class ProjectorPyGame(Projector):
    def __init__(self, display:int, homography_matrix = np.diag([1.0, 1.0, 1.0])):
        super().__init__(homography_matrix)
        self.resolution = None
        self.screen = None
        self.dot_radius = 20
        self.homography_matrix = homography_matrix
        self.display = display

    def __enter__(self):
        pygame.init()
        self.resolution = pygame.display.get_desktop_sizes()[self.display]
        display_info = pygame.display.get_desktop_sizes()
        self.screen = pygame.display.set_mode(display_info[self.display], pygame.FULLSCREEN, display=self.display)
        self.show_empty_view()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pygame.quit()

    def show_dot_in_projector_view(self, proj_pt: tuple[int, int], radius: int = None) -> None:
        if radius is None:
            radius = self.dot_radius
        self.show_empty_view()
        logging.info(f"Project Point at {proj_pt} with Dot Radius: {radius}")
        pygame.draw.circle(self.screen, (255, 0, 0), proj_pt, radius)
        pygame.display.flip()

    def show_empty_view(self):
        logging.info(f"Show Empty Window")
        self.screen.fill((255, 255, 255))
        pygame.display.flip()

    def show_dot_in_camera_view(self, cam_pt: tuple[int, int])->None:
        cam_point = np.array([cam_pt[0], cam_pt[1], 1.0])
        proj_point_hom = self.homography_matrix @ cam_point
        proj_point_hom /= proj_point_hom[2]
        proj_x :int = proj_point_hom[0].astype(int)
        proj_y : int = proj_point_hom[1].astype(int)
        logging.info(f"Show Dot In Camera View at camera position {cam_pt} and projection point ({proj_x, proj_y})")
        self.show_dot_in_projector_view((proj_x, proj_y))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with ProjectorPyGame(0) as p:
        p.show_dot_in_projector_view((200, 200))
        time.sleep(1)
        p.show_empty_view()
        time.sleep(1)
        p.show_dot_in_projector_view((300, 300))
        time.sleep(1)
        p.show_dot_in_projector_view((400, 400))
        time.sleep(1)
        p.show_dot_in_camera_view((500, 500))
        time.sleep(3)