

def findSurroundingGridPoints(corner_point, grid_coordinates):
    x, y = corner_point
    x_coords = sorted(set(p[0] for p in grid_coordinates))
    y_coords = sorted(set(p[1] for p in grid_coordinates))

    closest_left_x = max([gx for gx in x_coords if gx <= x], default=None)
    closest_right_x = min([gx for gx in x_coords if gx >= x], default=None)

    closest_top_y = max([gy for gy in y_coords if gy <= y], default=None)
    closest_bottom_y = min([gy for gy in y_coords if gy >= y], default=None)

    surrounding_points = [
        (closest_left_x, closest_bottom_y),
        (closest_left_x, closest_top_y),
        (closest_right_x, closest_bottom_y),
        (closest_right_x, closest_top_y),
    ]


    surrounding_points = [p for p in surrounding_points if None not in p]

    return surrounding_points

def calculate_ratios(x, y, surrounding_points):

    bottom_left = None
    bottom_right = None
    top_left = None
    top_right = None

    for px, py in surrounding_points:
        if px <= x and py <= y:
            top_left = (px, py)
        elif px >= x and py <= y:
            top_right = (px, py)
        elif px <= x and py >= y:
            bottom_left = (px, py)
        elif px >= x and py >= y:
            bottom_right = (px, py)

    if not (bottom_left and bottom_right and top_left and top_right):
        raise ValueError("Nicht alle vier umgebenden Punkte vorhanden.")


    x1, y1 = top_left
    x2, y2 = bottom_right

    x_ratio = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
    y_ratio = (y - y1) / (y2 - y1) if y2 != y1 else 0.0

    return {
        "x_ratio": x_ratio,
        "y_ratio": y_ratio
    }