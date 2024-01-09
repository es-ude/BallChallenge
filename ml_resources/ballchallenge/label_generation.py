import numpy as np


def _gaussian_density(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (
        1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (values - mean) ** 2 / std**2)
    )


def _gaussian_density_2d(
    x: np.ndarray, y: np.ndarray, mean_point: tuple[float, float], std: float
) -> np.ndarray:
    x_mean, y_mean = mean_point
    x_gauss = _gaussian_density(x, mean=x_mean, std=std)
    y_gauss = _gaussian_density(y, mean=y_mean, std=std)
    return x_gauss * y_gauss


def generate_smooth_labels(
    target_points: list[tuple[float, float]],
    target_std: float,
    grid_size: tuple[int, int],
) -> np.ndarray:
    labels = np.zeros((len(target_points), *grid_size), dtype=np.float32)

    x_range = np.arange(grid_size[0])
    y_range = np.arange(grid_size[1])
    x, y = np.meshgrid(x_range, y_range)

    for i, point in enumerate(target_points):
        labels[i] = _gaussian_density_2d(x, y, mean_point=point, std=target_std)

    return labels


def generate_hard_labels(
    target_points: list[tuple[float, float]],
    grid_size: tuple[int, int],
) -> np.ndarray:
    labels = np.zeros((len(target_points), *grid_size), dtype=np.float32)

    for i, point in enumerate(target_points):
        labels[i, *point[::-1]] = 1

    return labels
