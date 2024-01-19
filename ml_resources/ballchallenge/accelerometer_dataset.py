from collections.abc import Callable
from typing import Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd

from ballchallenge.label_generation import generate_smooth_labels, generate_hard_labels


def _load_samples_and_positions(
    dataset_root: Path, labels_file_name: str, clipped_sample_length: int
) -> tuple[torch.Tensor, list[tuple[float, float]]]:
    df_positions = pd.read_csv(dataset_root / labels_file_name)

    samples = []
    positions = []

    for record in df_positions.itertuples():
        positions.append((record.x_position, record.y_position))
        sample = pd.read_csv(dataset_root / record.file).values.T
        samples.append(torch.tensor(sample)[:, :clipped_sample_length])

    return torch.stack(samples), positions


class AccelerometerDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        grid_size: tuple[int, int] = (40, 40),
        x_position_range: tuple[float, float] = (0, 2),
        y_position_range: tuple[float, float] = (0, 2),
        label_std: float = 0.3,
        transform_samples: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform_labels: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self._samples, positions = _load_samples_and_positions(
            dataset_root=dataset_root,
            labels_file_name="labels.csv",
            clipped_sample_length=1000,
        )
        self._labels = torch.tensor(
            generate_smooth_labels(
                target_points=positions,
                target_std=label_std,
                x_value_range=x_position_range,
                y_value_range=y_position_range,
                grid_size=grid_size,
            )
        )

        if transform_samples is not None:
            self._samples = transform_samples(self._samples)

        if transform_labels is not None:
            self._labels = transform_labels(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self._samples[index], self._labels[index]
