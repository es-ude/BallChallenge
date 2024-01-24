import numpy as np
from math import sqrt


def _normal_density(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return 1 / (std * sqrt(2 * np.pi)) * np.exp(-0.5 * ((values - mean) / std) ** 2)


def _normal_density_2d(
    x: np.ndarray, y: np.ndarray, mean_point: tuple[float, float], std: float
) -> np.ndarray:
    x_gauss = _normal_density(values=x, mean=mean_point[0], std=std)
    y_gauss = _normal_density(values=y, mean=mean_point[1], std=std)
    return x_gauss * y_gauss


def generate_smooth_labels(
    target_points: list[tuple[float, float]],
    target_std: float,
    x_value_range: tuple[float, float],
    y_value_range: tuple[float, float],
    grid_size: tuple[int, int],
) -> np.ndarray:
    labels = np.zeros((len(target_points), *grid_size), dtype=np.float32)

    x_range = np.linspace(*x_value_range, grid_size[0])
    y_range = np.linspace(*y_value_range, grid_size[1])
    x, y = np.meshgrid(x_range, y_range)

    for i, point in enumerate(target_points):
        labels[i] = _normal_density_2d(x, y, mean_point=point, std=target_std)

    normalized_labels = labels / labels.sum()

    return normalized_labels


def generate_hard_labels(
    target_points: list[tuple[float, float]],
    target_std: float,
    x_value_range: tuple[float, float],
    y_value_range: tuple[float, float],
    grid_size: tuple[int, int],
) -> np.ndarray:  #
    labels = np.zeros((len(target_points), *grid_size), dtype=np.float32)
    for i, point in enumerate(target_points):
        x_diff = x_value_range[1] - x_value_range[0]
        y_diff = y_value_range[1] - y_value_range[0]
        x = point[0]
        y = point[1]
        num_x_sections = grid_size[0]
        num_y_sections = grid_size[1]
        x_grid_pos = int(x / x_diff * num_x_sections)
        y_grid_pos = int(y / y_diff * num_y_sections)

        labels[i, x_grid_pos, y_grid_pos] = 1

    return labels
