import logging
import time
from typing import Any

import cv2


class Camera:
    def __init__(self, camera_index = 0):
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(self.camera_index)
        self.resolution = tuple([self.camera.get(cv2.CAP_PROP_FRAME_WIDTH), self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.camera.release()

    def take_picture(self, trys_to_take_photo=5)->Any:
        for i in range(trys_to_take_photo):
            ret, frame = self.camera.read()
            if not ret:
                logging.error("Can't receive frame. Exiting ...")
                time.sleep(1)
                continue
            logging.info(f"Take Picture")
            return frame.copy()
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with Camera() as cam:
        print("go")
        pic = cam.take_picture()
        cv2.imshow('Camera View', pic)
        cv2.waitKey(0)
        pic = cam.take_picture()
        cv2.imshow('Camera View', pic)
        cv2.waitKey(0)
