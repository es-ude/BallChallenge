import cv2
import os
import json
import math
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from create_dynamic_grid import Vector, Line, Grid, Square, Circle

def filter_image_by_blobs(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 15
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.minConvexity = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(grey_image)

    blobs_coordinates = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return blobs_coordinates

def draw_image(image, points):
    keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in points]
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Image', image_with_keypoints)
    cv2.waitKey(0)

def convert_points_to_lines(edge_points):
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(edge_points)
    distances, neighbors = nn.kneighbors(edge_points)

    lines = []
    for i in range(len(edge_points)):
        p1 = edge_coordinates_as_points[i]
        p2 = edge_coordinates_as_points[neighbors[i, 1]]
        lines.append(Line(Vector(p1[0], p1[1]), Vector(p2[0], p2[1])))

    filtered_lines = []
    for i in lines:
        if i not in filtered_lines:
            filtered_lines.append(i)

    return filtered_lines

def draw_lines_on_image(image, lines):
    for i in lines:
        point_one = (int(i._a.get_x), int(i._a.get_y))
        point_two = (int(i._b.get_x), int(i._b.get_y))
        cv2.line(image, point_one, point_two, (0, 0, 255), 2)
    cv2.imshow("Lines", image)
    cv2.waitKey(0)

def improve_blobs(point_coords, eps=20, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_coords)
    labels = db.labels_
    filtered_blobs = point_coords[labels != -1]
    return filtered_blobs

def calculate_cornerpoints_for_grid_to_vector_coords(grid_points):
    min_x = np.min(grid_points[:, 0])
    max_x = np.max(grid_points[:, 0])
    min_y = np.min(grid_points[:, 1])
    max_y = np.max(grid_points[:, 1])

    down_left = Vector(min_x, max_y)
    down_right = Vector(max_x, max_y)
    up_left = Vector(min_x, min_y)
    up_right = Vector(max_x, min_y)

    corner_points = [down_left, down_right, up_left, up_right]
    return corner_points

def get_squares(grid_points):
    squares = []
    for j in range(int(math.sqrt(len(grid_points))) - 1):
        for k in range(int(math.sqrt(len(grid_points))) - 1):
            squares.append(Square([Line(Vector(grid_points[k + j * int(math.sqrt(len(grid_points)))][0], grid_points[k + j * int(math.sqrt(len(grid_points)))][1]),
                                        Vector(grid_points[k + j * int(math.sqrt(len(grid_points))) + 1][0], grid_points[k + j * int(math.sqrt(len(grid_points))) + 1][1])),
                                   Line(Vector(grid_points[k + j * int(math.sqrt(len(grid_points))) + 1][0], grid_points[k + j * int(math.sqrt(len(grid_points))) + 1][1]),
                                        Vector(grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points))) + 1][0], grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points))) + 1][1])),
                                   Line(Vector(grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points))) + 1][0], grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points))) + 1][1]),
                                        Vector(grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points)))][0], grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points)))][1])),
                                   Line(Vector(grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points)))][0], grid_points[k + j * int(math.sqrt(len(grid_points))) + int(math.sqrt(len(grid_points)))][1]),
                                        Vector(grid_points[k + j * int(math.sqrt(len(grid_points)))][0], grid_points[k][1]))]))
    return squares

def filter_edge_points(list_of_squares, list_of_points, radius=30):
    edge_coordinates = []
    for i in list_of_squares:
        for j in i.points_inside_square(convert_point_coordinates_to_vector_coordinates(list_of_points)):
            if Circle(j, radius=radius).amount_of_points_in_circle(convert_point_coordinates_to_vector_coordinates(list_of_points)) <= 5:
                edge_coordinates.append(j)
    return edge_coordinates

def cluster_lines_by_distance(lines_clustered_by_angle):
    certain_distance_group = defaultdict(list)
    distances_of_lines = []

    for i in lines_clustered_by_angle:
        distance = i.distance()
        distances_of_lines.append([distance])

    dbscan = DBSCAN(eps=3, min_samples=3)
    labels = dbscan.fit_predict(distances_of_lines)

    for i in range(len(labels)):
        if labels[i] != -1:
            certain_distance_group[labels[i]].append(lines_clustered_by_angle[i])

    filtered_lines = {label: lines for label, lines in certain_distance_group.items() if len(lines) > 4}
    return filtered_lines

def cluster_lines_by_angle(lines, tolerance=5):

    certain_angle_group = defaultdict(list)

    for line in lines:
        angle = line.calculate_angle_of_normal_vector()
        rounded_angle = round(angle / tolerance) * tolerance
        certain_angle_group[rounded_angle].append(line)

    filtered_groups = {angle: lines_sec for angle, lines_sec in certain_angle_group.items() if len(lines_sec) > 20}

    return filtered_groups

def convert_squares_in_coords(list_of_squares):
    corner_coords = []
    for i in list_of_squares:
        corner_coords.extend(i.get_corner_coords())
    return corner_coords

def convert_point_coordinates_to_vector_coordinates(points):
    points_as_vectors = []
    for i in points:
        points_as_vectors.append(Vector(i[0], i[1]))
    return points_as_vectors

def convert_vector_coordinates_to_point_coordinates(vectors):
    vectors_as_points = []
    for i in vectors:
        vectors_as_points.append((i.get_x, i.get_y))
    return vectors_as_points

def extract_items_from_dict(dictionary):
    items = []

    for label, group in dictionary.items():
        for i in group:
            items.append(i)

    return items

def connect_lines_of_each_edge(lines):

    connected_lines = []

    for distance, distance_groups in lines.items():
        vectors = []
        for i in distance_groups:
            vectors.append(i._a)
            vectors.append(i._b)
        coordinates_as_points = convert_vector_coordinates_to_point_coordinates(vectors)
        min_punkt = min(coordinates_as_points, key=lambda p: (p[0], p[1]))
        max_punkt = max(coordinates_as_points, key=lambda p: (p[0], p[1]))
        connected_lines.append(Line(Vector(min_punkt[0], min_punkt[1]), Vector(max_punkt[0], max_punkt[1])))
    return connected_lines

def extract_corner_points(edge_lines, tolerance=20):
    corner_points = []
    for i in range(0, len(edge_lines)):
        for j in range(0 + i, len(edge_lines)):
            if not single_edge_lines[i].is_parallel(edge_lines[j], tolerance):
                corner_points.append(edge_lines[i].intersection(edge_lines[j]))
    return corner_points

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, "config.json"), "r") as file:
        config = json.load(file)

    ORIGINAL_IMAGE_PATH = os.path.join(BASE_DIR, config["edge_image"])

    new_image = cv2.imread(ORIGINAL_IMAGE_PATH)

    blob_coords = filter_image_by_blobs(new_image)

    filtered_blob_coords = improve_blobs(blob_coords)

    corner_points = calculate_cornerpoints_for_grid_to_vector_coords(filtered_blob_coords)

    grid_vectors = Grid([Line(corner_points[0], corner_points[2]), Line(corner_points[2], corner_points[3]), Line(corner_points[3], corner_points[1]), Line(corner_points[1], corner_points[0])], 15).calculate_grid_points()

    grid_points = convert_vector_coordinates_to_point_coordinates(grid_vectors)

    sorted_grid_points = sorted(grid_points, key=lambda p: (p[0], p[1]))

    squares_in_list = get_squares(sorted_grid_points)

    square_coords = convert_squares_in_coords(squares_in_list)

    edge_coordinates_as_vectors = filter_edge_points(squares_in_list, filtered_blob_coords)

    edge_coordinates_as_points = convert_vector_coordinates_to_point_coordinates(edge_coordinates_as_vectors)

    edge_lines = convert_points_to_lines(edge_coordinates_as_points)

    clustered_lines_by_angle = cluster_lines_by_angle(edge_lines)

    extracted_lines = extract_items_from_dict(clustered_lines_by_angle)

    clustered_lines_by_distance = cluster_lines_by_distance(extracted_lines)

    single_edge_lines = connect_lines_of_each_edge(clustered_lines_by_distance)

    corner_points_as_vectors = extract_corner_points(single_edge_lines)

    corner_coordinates_as_points = convert_vector_coordinates_to_point_coordinates(corner_points_as_vectors)

    draw_image(new_image, corner_coordinates_as_points)
















