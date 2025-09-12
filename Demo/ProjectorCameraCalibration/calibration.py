import datetime
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from Demo.ProjectorCameraCalibration.camera import Camera
from Demo.ObjectDetection.circle_detection import circle_detection_for_picture_differences
from Demo.ProjectorCameraCalibration.projector_pygame import ProjectorPyGame


class CameraProjectorCalibration:
    def __init__(self, camera: Camera, projector: ProjectorPyGame):
        self._camera = camera
        self._projector = projector
        self.H_cam_to_proj = None
        self.H_proj_to_cam = None
        self.calibration_point_factors = (0.35, 0.65)
        self.sleep_time_projector_to_camera = 1
        self.sleep_time_camera_to_projector = 1
        self.debug = True
        self.debug_object_dection = False
        time_now = "log_{}".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.debug_folder = Path("debug").joinpath(time_now)
        self.debug_folder.mkdir(parents=True, exist_ok=True)
        self.debug_try_counter = 0
        logging.basicConfig(filename=self.debug_folder.joinpath("log.txt"), level=logging.DEBUG)

    def calibrate(self, num_points = 20) -> tuple[np.ndarray, np.ndarray] | None:
        def clean_not_detected_circles(points_for_projector_in_projector_view: list[tuple[int, int]], points_detected_by_camera_in_camera_view: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int,int]]]:
            points_for_projector_in_projector_view_cleaned = []
            points_detect_by_camera_in_camera_view_cleaned = []
            for i in range(len(points_for_projector_in_projector_view)):
                if points_detected_by_camera_in_camera_view[i] == (-1, -1):
                    continue
                points_for_projector_in_projector_view_cleaned.append(points_for_projector_in_projector_view[i])
                points_detect_by_camera_in_camera_view_cleaned.append(points_detected_by_camera_in_camera_view[i])
            return points_for_projector_in_projector_view_cleaned, points_detect_by_camera_in_camera_view_cleaned
        points_for_projector_in_projector_view: list[tuple[int,int]] = self._gen_points_to_display_in_projector_view(num_points)
        points_detected_by_camera_in_camera_view = self.project_points_and_detect_them_with_object_detection_and_camera(points_for_projector_in_projector_view, False)
        points_for_projector_in_projector_view_cleaned, points_detect_by_camera_in_camera_view_cleaned = clean_not_detected_circles(points_for_projector_in_projector_view,points_detected_by_camera_in_camera_view)
        if len(points_for_projector_in_projector_view_cleaned) >= 4:
            self.H_proj_to_cam, self.H_cam_to_proj = compute_homographies(points_for_projector_in_projector_view_cleaned, points_detect_by_camera_in_camera_view_cleaned)
            self._camera.homography_matrix = self.H_proj_to_cam
            self._projector.homography_matrix = self.H_cam_to_proj
            return (self.H_proj_to_cam, self.H_cam_to_proj)
        else:
            return None

    def project_points_and_detect_them_with_object_detection_and_camera(self, points: list[tuple[int, int]], show_points_in_camera_view: bool) -> list[tuple[int, int]]:
        points_detected_by_camera_in_camera_view: list[tuple[int, int]] = []
        for i, point in enumerate(points):
            logging.info(f"{i} Point")
            point = self.detect_point_with_diff_pictures(point, show_points_in_camera_view)
            points_detected_by_camera_in_camera_view.append(point)
            logging.info(f"Have detected dot in {points_detected_by_camera_in_camera_view[-1]}")
        self.debug_try_counter += 1
        logging.debug(f"{self.debug_try_counter}")
        return points_detected_by_camera_in_camera_view

    def detect_point_with_diff_pictures(self, point: tuple[int, int], show_dot_in_camera_view: bool) -> tuple[int, int]:
        self._projector.show_empty_view()
        time.sleep(self.sleep_time_projector_to_camera)
        pic_0 = self._camera.take_picture()
        time.sleep(self.sleep_time_camera_to_projector)
        if show_dot_in_camera_view:
            self._projector.show_dot_in_camera_view(point)
        else:
            self._projector.show_dot_in_projector_view(point)
        time.sleep(self.sleep_time_projector_to_camera)
        pic_1 = self._camera.take_picture()
        time.sleep(self.sleep_time_camera_to_projector)
        if self.debug:
            if show_dot_in_camera_view:
                name = "camera_view"
            else:
                name = "projection_view"
            self.debug_folder.joinpath(f"try_{self.debug_try_counter}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(self.debug_folder.joinpath(f"try_{self.debug_try_counter}").joinpath(f"pic_{name}_{point[0]}_ {point[1]}_empty.png"), pic_0)
            cv2.imwrite(self.debug_folder.joinpath(f"try_{self.debug_try_counter}").joinpath(f"pic_{name}_{point[0]}_ {point[1]}_with.png"), pic_1)
        return circle_detection_for_picture_differences(pic_0, pic_1, show_image=self.debug_object_dection)

    def get_calibration_error(self, num_points = 20) -> list[tuple[int, int]]:
        points_for_projector_in_camera_view: list[tuple[int,int]] = self._gen_points_to_display_in_camera_view(num_points)
        points_detected_by_camera_in_camera_view = self.project_points_and_detect_them_with_object_detection_and_camera(points_for_projector_in_camera_view, True)
        error: list[tuple[int, int]] = []
        for i in range(len(points_for_projector_in_camera_view)):
            if points_detected_by_camera_in_camera_view[i] == (-1, -1):
                continue
            x_error = points_for_projector_in_camera_view[i][0] - points_detected_by_camera_in_camera_view[i][0]
            y_error = points_for_projector_in_camera_view[i][1] - points_detected_by_camera_in_camera_view[i][1]
            error.append((x_error, y_error))
        if len(error) == 0:
            logging.error("No points detected. Calibration did not work")
            return error
        return error

    def _gen_points_to_display_in_camera_view(self, num_points: int) -> list[tuple[int,int]]:
        return self._gen_points_for_view(num_points, self._camera.resolution)

    def _gen_points_to_display_in_projector_view(self, num_points: int) -> list[tuple[int,int]]:
        return self._gen_points_for_view(num_points, self._projector.resolution)

    def _gen_points_for_view(self, num_points: int, resolution: tuple[int, int]) -> list[tuple[int,int]]:
        num_points_per_axis = math.ceil(math.sqrt(num_points))
        x, y = resolution
        x_min = x*self.calibration_point_factors[0]
        x_max = x*self.calibration_point_factors[1]
        y_min = y*self.calibration_point_factors[0]
        y_max = y*self.calibration_point_factors[1]
        xs = np.linspace(x_min, x_max, num=num_points_per_axis)
        ys = np.linspace(y_min, y_max, num=num_points_per_axis)
        points = []
        for i in range(num_points_per_axis):
            for j in range(num_points_per_axis):
                points.append((int(xs[i].astype(np.int64)), int(ys[j].astype(np.int64))))
        return points

def compute_homographies(projector_points: list[tuple[int, int]], camera_points: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    proj_pts = np.array(projector_points, dtype=np.float32)
    cam_pts = np.array(camera_points, dtype=np.float32)
    if len(proj_pts) < 4 or len(cam_pts) < 4:
        raise ValueError("Need at least 4 valid point correspondences to compute homography.")
    H_proj_to_cam, _ = cv2.findHomography(proj_pts, cam_pts, method=cv2.RANSAC)
    H_cam_to_proj, _ = cv2.findHomography(cam_pts, proj_pts, method=cv2.RANSAC)
    logging.info(f"H_proj_to_cam: {H_proj_to_cam}")
    logging.info(f"H_cam_to_proj: {H_cam_to_proj}")
    return H_proj_to_cam, H_cam_to_proj


