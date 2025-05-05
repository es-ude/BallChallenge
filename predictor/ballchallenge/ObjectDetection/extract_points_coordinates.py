import cv2
import os
import json
import numpy as np

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, "config.json"), "r") as file:
        config = json.load(file)

    CURRENT_IMAGE = os.path.join(BASE_DIR, config["extract_coordinates_image"])

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Bild", img)

if __name__ == "__main__":
    img = cv2.imread(CURRENT_IMAGE)
    points = []


    cv2.imshow("Bild", img)
    cv2.setMouseCallback("Bild", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(points)
