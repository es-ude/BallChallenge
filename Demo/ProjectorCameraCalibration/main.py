import logging
from pathlib import Path

import cv2

from Demo.ProjectorCameraCalibration.calibration import CameraProjectorCalibration
from Demo.ProjectorCameraCalibration.camera import Camera
from Demo.ProjectorCameraCalibration.projector_pygame import ProjectorPyGame

def calculate_mean_error(errors: list[tuple[int, int]]) -> tuple[int, int]:
    if len(errors) > 0:
        x = 0
        y = 0
        for error in errors:
            x += error[0]
            y += error[1]
        x /= len(errors)
        y /= len(errors)
        return x, y
    else:
        return 9999999,99999999

def check_if_error_smaller_5px_mean(mean_error: tuple[int, int]) -> bool:
    allowed_error = 5
    if mean_error[0] < allowed_error and mean_error[1] < allowed_error:
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Calibration")
    monitor_id = 1
    with Camera() as camera:
        with ProjectorPyGame(monitor_id) as projector:
            projector.load_homography_matrix(Path("Homography.csv"))
            cpc = CameraProjectorCalibration(camera, projector)
            cpc.debug = True
            num_points_per_axis = 3
            errors = cpc.get_calibration_error(num_points_per_axis ** 2)
            mean_error = calculate_mean_error(errors)
            logging.debug(f"Mean error: {errors}")
            logging.info(f"Calibration error: {mean_error}")
            max_trys_for_calibration = 6
            trys_for_calibration = 0
            while not check_if_error_smaller_5px_mean(mean_error):
                logging.info("Calibration errortoo big. Rerun Calibration.")
                num_points_per_axis += 1
                logging.info(f"num points: {num_points_per_axis ** 2}")
                status = cpc.calibrate(num_points_per_axis ** 2)
                if status is None:
                    if max_trys_for_calibration < max_trys_for_calibration:
                        continue
                    break
                errors = cpc.get_calibration_error(num_points_per_axis ** 2)
                mean_error = calculate_mean_error(errors)
                logging.debug(f"Mean error: {errors}")
                logging.info(f"Calibration error: {mean_error}")
                trys_for_calibration += 1
            projector.save_homography_matrix(Path("Homography.csv"))
            logging.info("System Calibrated")
