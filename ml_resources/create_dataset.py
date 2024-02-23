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

TOO_SMALL = 50


def order_points(pts):
    try:
        pts = np.reshape(pts, (4, 2))
    except:
        raise ValueError("No rect found")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
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


def find_field(image: Path) -> list[tuple[float, float]]:
    img = cv2.imread(str(image))

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    df = pd.DataFrame(
        hierarchy[0],
        columns=["nextSameLevel", "prviousSameLevel", "FirstChild", "Parent"],
    )
    fieldIdx = df.groupby("Parent").count().idxmax().nextSameLevel
    field = contours[fieldIdx]
    x, y, w, h = cv2.boundingRect(field)
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    epsilon = 0.1 * cv2.arcLength(field, True)
    approx = cv2.approxPolyDP(field, epsilon, True)
    rect2 = order_points(approx)

    m = cv2.getPerspectiveTransform(rect2, rect)

    dst = cv2.warpPerspective(img, m, (int(w), int(h)))
    dst = cv2.resize(dst, (int(w), int(w)), cv2.INTER_AREA)

    imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 140, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    field_width = w
    sacks = fitler_sacks(contours)
    positions = []

    for sack in sacks:
        x, y, w, h = cv2.boundingRect(sack)
        center = (x + w / 2, y + h / 2)
        positions.append((center[0] / field_width * 2, center[1] / field_width * 2))

    return positions


def extract_samples_and_labels(raw_dataset: Path, destination: Path) -> None:
    trials = filter(lambda f: f.is_dir(), sorted(raw_dataset.glob("*")))

    sample_files: list[Path] = []
    labels: list[tuple[float, float]] = []

    for trial in trials:
        image_file = trial / "image.jpg"
        sample_file = trial / "measurement.csv"

        positions = find_field(image_file)
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
        raw_dataset=args.raw_dataset, destination=args.destination
    )


if __name__ == "__main__":
    main()
