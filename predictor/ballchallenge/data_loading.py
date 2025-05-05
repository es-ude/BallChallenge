from pathlib import Path
from typing import cast

import torch
from torch.utils.data import Dataset, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT = Path("./data")


def split_dataset(ds: Dataset) -> tuple[Dataset, Dataset]:
    return tuple(random_split(ds, lengths=[0.75, 0.25], generator=torch.Generator().manual_seed(42)))


def get_input_data_shape(ds: Dataset) -> tuple[int, int]:
    return cast(tuple[int, int], tuple(ds[:][0].shape[1:]))


def flat_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.flatten(start_dim=1)


def downsample(samples: torch.Tensor, factor: int) -> torch.Tensor:
    return samples[:, :, ::factor]


def downsample_and_substract_mean(samples: torch.Tensor, factor):
    return samples[:, :, ::factor] - samples[:, :, ::factor].mean(dim=2, keepdim=True)


