"""
Note: This code uses code written by Dr.-Ing. Viktor Matkovic (https://www.vs.uni-due.de/person/matkovic/).
"""

from pathlib import Path
from argparse import ArgumentParser
from shutil import copyfile
import csv

import cv2
import numpy as np
import pandas as pd
import random

TOO_SMALL = 50


def convert_points_to_rect(pts):
    try:
        pts = np.reshape(pts, (4, 2))
    except:
        raise ValueError("No rect found")

    rect = np.zeros((4, 2), dtype="float32")
    distances_from_zero = np.linalg.norm(pts, ord=2, axis=1)
    print(pts)
    s = pts.sum(axis=1)
    id_of_a = np.argmin(distances_from_zero)
    vec_to_a = pts[np.argmin(distances_from_zero)]
    normed_a = vec_to_a / distances_from_zero[id_of_a]
    normed_vectors = tuple(p /d for p,d in zip(pts, distances_from_zero))
    print(normed_vectors)
    projections_on_vec_to_a = tuple(vec_to_a @ v for v in normed_vectors)
    print(projections_on_vec_to_a)
    id_of_corner_d = np.argmin(projections_on_vec_to_a)
    corner_d = pts[id_of_corner_d]
    id_of_corner_c = np.argmax(projections_on_vec_to_a)
    corner_c = pts[id_of_corner_c]
    corner_b = vec_to_a + corner_d - corner_c
    print(vec_to_a, corner_b, corner_c, corner_d)
    return np.array((vec_to_a, corner_b, corner_c, corner_d)).reshape((4, 2))


def convert_points_to_cv2_rectangle(pts):
    try:
        pts = np.reshape(pts, (4, 2))
    except:
        raise ValueError("No rect found")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    pt1 = pts[np.argmin(s)]
    pt2 = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    width = pts[np.argmin(diff)]
    height = pts[np.argmax(diff)]
    rect[0] = pt1
    rect[1] = width
    rect[2] = pt2
    rect[3] = height
    return rect


def fitler_sacks(contours):
    sacks = list()
    mean = 0
    for index in range(1, len(contours)):
        cnt = contours[index]
        area = cv2.contourArea(cnt)
        mean = (mean + area) / 2

        if index == 1:
            mean = area

        if area <= mean / 2:
            break

        if area < TOO_SMALL:
            break

        sacks.append(cnt)

    return sacks

def find_contour_that_is_most_often_referenced_as_parent_in_hierarchy(contours, hierarchy):
    parents = hierarchy[0, :, 3]
    contour_ids = pd.DataFrame(
            {'parent': parents, 'child': np.arange(parents.shape[0])}
    )
    rows_sorted = list(contour_ids.groupby("parent").count().sort_values(by="child", ascending=False).iterrows())
    id  = 0
    for parent_id, _ in rows_sorted[1:]:
        if parent_id > 0:
            id = parent_id
            break

    contour = contours[id]
    return contour


class PlayingField:
    def __init__(self, original_rectangle, normalized_rectangle):
        self.original_rectangle = original_rectangle
        self.normalized_rectangle = normalized_rectangle

    @property
    def width(self) -> int:
        return int(self.normalized_rectangle[2][0])

    @property
    def height(self) -> int:
        return int(self.normalized_rectangle[2][1])

    def normalize_image(self, img):
        m = cv2.getPerspectiveTransform(self.normalized_rectangle, self.original_rectangle)

        dst = cv2.warpPerspective(img, m, (self.width, self.height))
        dst = cv2.resize(dst, (self.width, self.height), cv2.INTER_AREA)
        return dst

    @staticmethod
    def from_image(img):

        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        def rand_color():
            return (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        
        def draw_contour(contour):
            drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(drawing, [contour], 0, color=rand_color(), thickness=2)
            cv2.imshow('Contours', drawing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def draw_line(points):
            drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
            c = rand_color()
            for p1, p2 in zip(points[:-1], points[1:]):
                cv2.line(drawing, p1, p2, c, thickness=2)
            cv2.imshow("im", drawing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        field_contour = find_contour_that_is_most_often_referenced_as_parent_in_hierarchy(contours, hierarchy)

        x, y, w, h = cv2.boundingRect(field_contour)


        rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)  

        def approximate_with_lower_poly_line(contour, factor):
            epsilon = factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return approx
        
        def turn_contour_into_warped_rectangle(field_contour):
            approx = approximate_with_lower_poly_line(field_contour, 0.03)
            approx = cv2.convexHull(approx)
            approx = approximate_with_lower_poly_line(approx, 0.03)
            rect2 = convert_points_to_cv2_rectangle(approx)
            return rect2

        rect2 = turn_contour_into_warped_rectangle(field_contour)
        return PlayingField(rect2, rect)


def get_sack_positions(field: PlayingField, image: Path):
    image = cv2.imread(str(image))
    dst = field.normalize_image(image)
    imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 140, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sacks = fitler_sacks(contours)
    positions = []

    for sack in sacks:
        x, y, w, h = cv2.boundingRect(sack)
        center = (x + w / 2, y + h / 2)
        positions.append((center[0] / field.width * 2, center[1] / field.width * 2))

    return positions


def extract_samples_and_labels(raw_dataset: Path, destination: Path, calibration_image: str) -> None:
    trials = filter(lambda f: f.is_dir(), sorted(raw_dataset.glob("*")))

    sample_files: list[Path] = []
    labels: list[tuple[float, float]] = []
    field = PlayingField.from_image(cv2.imread(calibration_image))

    for trial in trials:
        image_file = trial / "image.jpg"
        sample_file = trial / "measurement.csv"

        positions = get_sack_positions(image=image_file, field=field)
        if len(positions) == 0:
            print(f"[ERROR] Cannot extract label from image ({image_file}).")
        elif len(positions) > 1:
            print(
                f"[ERROR] Cannot extract unambiguous label from image ({image_file})."
            )
        else:
            sample_files.append(sample_file)
            labels.append(positions[0])

    destination.mkdir(exist_ok=True)

    with (destination / "labels.csv").open("w") as out_file:
        label_file_writer = csv.writer(out_file)
        label_file_writer.writerow(["file", "x_position", "y_position"])

        for sample_file, label in zip(sample_files, labels):
            new_sample_name = f"{sample_file.parent.name}.csv"
            copyfile(sample_file, destination / new_sample_name)
            label_file_writer.writerow([new_sample_name, *label])


def main() -> None:
    parser = ArgumentParser(
        prog="Create Dataset",
        description="Extracts samples and labels from raw measurements.",
    )
    parser.add_argument("--raw_dataset", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()

    extract_samples_and_labels(
        raw_dataset=args.raw_dataset, destination=args.destination, calibration_image="Ball Throw Challenge/Measurements_25.01/SensorValues/2024-01-25 12:41:16.593/image.jpg"
    )


if __name__ == "__main__":
    main()