from typing import Any
import torch
from torch.utils.data import Dataset

from .label_generation import generate_smooth_labels


def _generate_dummy_samples_labels(
    grid_size: tuple[int, int] = (10, 10),
    sample_length: int = 1000,
    samples_per_pixel: int = 50,
    base_freq: int = 100,
    freq_offset: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    def sinus(frequency: float) -> torch.Tensor:
        xs = torch.linspace(0, 1, sample_length)
        return torch.sin(2 * torch.pi * frequency * xs)

    def offset(pixel: int) -> float:
        def noise() -> float:
            return float(torch.randn(1).item()) * freq_offset * 0.5

        return freq_offset * pixel + noise()

    samples = []
    target_positions = []

    x_size, y_size = grid_size
    for y_pixel in range(y_size):
        for x_pixel in range(x_size):
            for _ in range(samples_per_pixel):
                axis1 = sinus(base_freq + offset(x_pixel))
                axis2 = sinus(base_freq + offset(y_pixel))
                axis3 = torch.randn(sample_length)

                samples.append(torch.stack([axis1, axis2, axis3]))
                target_positions.append((x_pixel, y_pixel))

    samples = torch.stack(samples)
    labels = torch.tensor(generate_smooth_labels(target_positions, 1, grid_size))

    return samples, labels


class DummyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self._samples, self._labels = _generate_dummy_samples_labels(sample_length=1000)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self._samples[index], self._labels[index]
