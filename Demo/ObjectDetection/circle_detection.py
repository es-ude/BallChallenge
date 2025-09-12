import logging

import cv2

from Demo.ObjectDetection.object_detection import calculate_grayscale_diff, min_max_normalise_grayscale_diff_pic, \
    grayscale_object_detection


def circle_detection_for_picture_differences(image1, image2, show_image=False) -> tuple[int, int]:
    diff_image = calculate_grayscale_diff(image1, image2)
    min_max_diff = min_max_normalise_grayscale_diff_pic(diff_image, show_image=show_image)
    dot_position = grayscale_object_detection(min_max_diff, morph_kernel=(7,7), adaptive_threshold_blocksize=31, adaptive_threshold_c=10, area_min=1000, area_max=4000 , show_image=show_image)
    return dot_position


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pic_0 = cv2.imread("/Users/leo/work/BallChallenge/Demo/ProjectorCameraCalibration/debug/log_2025_09_24_10_03_47/pic_512_ 320_empty.png")
    pic_1 = cv2.imread("/Users/leo/work/BallChallenge/Demo/ProjectorCameraCalibration/debug/log_2025_09_24_10_03_47/pic_512_ 320_with.png")
    #image = calculate_red_color_diff(pic_0, pic_1 )
    #image = min_max_normalise_grayscale_diff_pic(image, show_image=True)
    circle_detection_for_picture_differences(pic_0, pic_1, show_image=True)