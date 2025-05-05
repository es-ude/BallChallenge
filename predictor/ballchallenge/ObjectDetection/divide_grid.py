import image_processing
from certain_grid_config import CertainGrid

if __name__ == "__main__":
        config = CertainGrid("config.json")


def intersection(x_BB_top_left, y_BB_top_right, x_BB_bottom_right, y_BB_bottom_right, x_grid_top_left, y_grid_top_left, x_grid_bottom_right, y_grid_bottom_right):
        x_overlap = max(0, min(x_BB_bottom_right, x_grid_bottom_right) - max(x_BB_top_left, x_grid_top_left))
        y_overlap = max(0, min(y_BB_bottom_right, y_grid_bottom_right) - max(y_BB_top_right, y_grid_top_left))
        return x_overlap * y_overlap


def divide_grid(corner_points, img):
    x_mid = ((corner_points[1])[0] - (corner_points[0])[0])/2
    y_mid = ((corner_points[3])[1] - (corner_points[0])[1])/2

    x, y, w, h = image_processing.process_image(img)

    areas = {"TopLeft": 0, "TopRight": 0, "ButtomLeft": 0, "ButtomRight": 0}
    areas["TopLeft"] = intersection(x - (corner_points[0])[0], y - (corner_points[0])[1], x + w - (corner_points[0])[0], y + h - (corner_points[0])[1], 0, 0, x_mid, y_mid)
    areas["TopRight"] = intersection(x - (corner_points[0])[0], y - (corner_points[0])[1], x + w - (corner_points[0])[0], y + h - (corner_points[0])[1], x_mid, 0, (config.get_corner_points()["TOP_RIGHT"])[0] - (config.get_corner_points()["TOP_LEFT"])[0], y_mid)
    areas["BottomLeft"] = intersection(x - (corner_points[0])[0], y - (corner_points[0])[1], x + w - (corner_points[0])[0], y + h - (corner_points[0])[1], 0, y_mid, x_mid, (config.get_corner_points()["BOTTOM_RIGHT"])[1] - (config.get_corner_points()["TOP_LEFT"])[1])
    areas["BottomRight"] = intersection(x - (corner_points[0])[0], y - (corner_points[0])[1], x + w - (corner_points[0])[0], y + h - (corner_points[0])[1], x_mid, y_mid, (config.get_corner_points()["BOTTOM_RIGHT"])[0] - (config.get_corner_points()["TOP_LEFT"])[0], (config.get_corner_points()["BOTTOM_RIGHT"])[1] - (config.get_corner_points()["TOP_LEFT"])[1])


    max_quadrant = max(areas, key=areas.get)

    return max_quadrant




