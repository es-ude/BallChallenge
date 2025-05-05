from functools import partial

import seaborn as sns
from elasticai.creator.nn import Sequential
from matplotlib import pyplot as plt
from torch import softmax

from ballchallenge.accelerometer_dataset import AccelerometerDataset
from ballchallenge.model_builder import ModelBuilder
from ballchallenge.data_loading import DATASET_ROOT, downsample_and_substract_mean, flat_labels


def render_heat_map_target_and_prediction(model, sample, target, grid_size):
    prediction = softmax(model(sample).detach(), dim=1).view(*grid_size[::-1])
    target = target.view(*grid_size[::-1])
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.heatmap(prediction, cmap="hot", square=True, ax=axs[0])
    axs[0].set_title("Prediction")
    sns.heatmap(target, cmap="hot", square=True, ax=axs[1])
    axs[1].set_title("Target")


def create_heat_map_model(input_shape: tuple[int, int], grid_size: tuple[int, int]) -> Sequential:
    model_builder = ModelBuilder(total_bits=8, frac_bits=4, input_shape=input_shape)

    model_builder.add_conv1d(filters=1, kernel_size=64).add_hard_tanh()
    model_builder.add_conv1d(filters=1, kernel_size=64).add_hard_tanh()
    model_builder.add_conv1d(filters=1, kernel_size=96).add_hard_tanh()
    model_builder.add_flatten()
    model_builder.add_linear(output_units=grid_size[0] * grid_size[1])

    return model_builder.build_model()


def load_heatmap_dataset(grid_size: tuple[int, int]) -> AccelerometerDataset:
    return AccelerometerDataset(
        dataset_root=DATASET_ROOT,
        grid_size=grid_size,
        x_position_range=(0, 2),
        y_position_range=(0, 2),
        label_std=0.3,
        transform_samples=partial(downsample_and_substract_mean, factor=4),
        transform_labels=flat_labels,
    )
