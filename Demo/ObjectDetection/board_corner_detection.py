import logging

import cv2
import numpy as np

class object_detection:


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pic_0 = cv2.imread("/Users/leo/work/BallChallenge/Demo/ProjectorCameraCalibration/debug/log_2025_09_24_10_03_47/pic_512_ 320_empty.png")
    pic_0 = cv2.cvtColor(pic_0, cv2.COLOR_BGR2GRAY)
    grayscale_object_detection(pic_0, True)