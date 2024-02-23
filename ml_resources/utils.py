import itertools
from functools import partial
from math import sqrt
from pathlib import Path
from typing import cast, Optional

import matplotlib.pyplot as plt
import torch
from elasticai.creator.nn import Sequential
from torch import softmax
from torch.nn import Sequential as torchSequential
from torch.utils.data import Dataset, random_split, DataLoader
import seaborn as sns
import numpy as np
from ballchallenge.accelerometer_dataset import AccelerometerDataset, AccelerometerDatasetWithPointLabels, \
    make_functions_for_normalizing_and_denormalizing_positions
from ballchallenge.model_builder import ModelBuilder
from ballchallenge.training import run_training_for_position, LossMetric, create_train_steps_factory, \
    create_test_steps_factory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT = Path("../data")


def split_dataset(ds: Dataset) -> tuple[Dataset, Dataset]:
    return tuple(random_split(ds, lengths=[0.75, 0.25], generator=torch.Generator().manual_seed(42)))


def get_input_data_shape(ds: Dataset) -> tuple[int, int]:
    return cast(tuple[int, int], tuple(ds[:][0].shape[1:]))


def plot_training_history(history) -> None:
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.plot(history.train["epoch"], history.train["loss"], label="train")
    ax.plot(history.test["epoch"], history.test["loss"], label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train History")
    ax.legend()


def create_heat_map_model(input_shape: tuple[int, int], grid_size: tuple[int, int]) -> Sequential:
    model_builder = ModelBuilder(total_bits=8, frac_bits=4, input_shape=input_shape)

    model_builder.add_conv1d(filters=1, kernel_size=64).add_hard_tanh()
    model_builder.add_conv1d(filters=1, kernel_size=64).add_hard_tanh()
    model_builder.add_conv1d(filters=1, kernel_size=96).add_hard_tanh()
    model_builder.add_flatten()
    model_builder.add_linear(output_units=grid_size[0] * grid_size[1])

    return model_builder.build_model()


def create_position_model(input_shape: tuple[int, int], kernel_size=64, hidden_channels=1,
                          num_convs=3) -> torch.nn.Module:
    act = torch.nn.ReLU
    flatten = torch.nn.Flatten
    linear = torch.nn.Linear
    in_length = input_shape[1]
    in_features = hidden_channels * (in_length + num_convs * (1 - kernel_size))
    conv = partial(torch.nn.Conv1d, in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
    layers: list[torch.nn.Module] = []
    for i in range(num_convs):
        if i == 0:
            layers.append(conv(in_channels=3))
        else:
            layers.append(conv())
        layers.append(torch.nn.BatchNorm1d(hidden_channels))
        layers.append(act())
    layers.append(flatten())
    layers.append(linear(in_features=in_features, out_features=2))
    layers.append(torch.nn.Tanh())
    return torchSequential(
        *tuple(layers)
    )


def flat_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.flatten(start_dim=1)


def downsample(samples: torch.Tensor, factor: int) -> torch.Tensor:
    return samples[:, :, ::factor]


def downsample_and_substract_mean(samples: torch.Tensor, factor):
    return samples[:, :, ::factor] - samples[:, :, ::factor].mean(dim=2, keepdim=True)


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


def load_point_dataset(xy_label_range: tuple[int, int]):
    return AccelerometerDatasetWithPointLabels(
        dataset_root=DATASET_ROOT,
        transform_samples=partial(downsample_and_substract_mean, factor=4),
    )


def render_heat_map_target_and_prediction(model, sample, target, grid_size):
    prediction = softmax(model(sample).detach(), dim=1).view(*grid_size[::-1])
    target = target.view(*grid_size[::-1])
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.heatmap(prediction, cmap="hot", square=True, ax=axs[0])
    axs[0].set_title("Prediction")
    sns.heatmap(target, cmap="hot", square=True, ax=axs[1])
    axs[1].set_title("Target")


def collect_prediction_target(model, ds):
    data = {'prediction': [], 'target': []}
    for id, (s, t) in enumerate(ds):
        p = model(s.view((1, s.size()[0], s.size()[1])))[0].detach().numpy()

        t = t.detach().numpy()

        data['prediction'].append(p)
        data['target'].append(t)
    return data


class PositionPrediction:
    def __init__(self, grid_size: tuple[int, int]):
        self.grid_size = grid_size
        self.model : Optional[torch.nn.Module] = None
        self.full_set = None
        self.train_set = None
        self.test_set = None
        self.history = None
        self.xy_label_range = (0, 2)
        self.move_positions_back = None
        self.train_loss, self.test_loss = LossMetric(), LossMetric()

    @property
    def input_shape(self):
        return get_input_data_shape(self.full_set)

    def load_dataset(self):
        move_positions_to_normalized_playing_field, move_positions_back =\
            make_functions_for_normalizing_and_denormalizing_positions(
                lower_left_corner=(0, 0),
                upper_right_corner=(2, 2))
        self.move_positions_back = move_positions_back
        self.full_set = AccelerometerDatasetWithPointLabels(
            dataset_root=DATASET_ROOT,
            transform_samples=partial(downsample_and_substract_mean, factor=4),
            transform_labels=move_positions_to_normalized_playing_field,
        )
        self.train_set, self.test_set = split_dataset(self.full_set)

    def _should_stop_early(self, loss_history: list) -> bool:
        if len(loss_history) > 20:
            loss_diff = tuple(l2 - l1 for l1, l2 in zip(loss_history[-21:-1], loss_history[-20:]))
            mean_diff = sum(loss_diff)/len(loss_diff)
            return mean_diff >= 0
        return False

    def train(self, epochs=200, batch_size=None, stop_early=True, lr=1e-4, quiet=False):
        if batch_size is None:
            batch_size = len(self.train_set)
        training_data = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(self.test_set, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_loss, self.test_loss = LossMetric(), LossMetric()
        train_loss, test_loss = self.train_loss, self.test_loss
        train_steps = create_train_steps_factory(self.model, training_data, DEVICE, torch.nn.functional.mse_loss, optimizer)
        test_steps = create_test_steps_factory(self.model, test_data, DEVICE, torch.nn.functional.mse_loss)
        best_model_state = self.model.state_dict()
        for epoch in range(epochs):
            for step_result in train_steps():
                train_loss.update(step_result.loss)
            for step_result in test_steps():
                test_loss.update(step_result.loss)
            if not quiet:
                print("epoch: {}, train loss: {}, test loss: {}".format(epoch, train_loss.value, test_loss.value))
            test_loss.reset()
            train_loss.reset()
            if min(test_loss.history) == test_loss.history[-1]:
                best_model_state = self.model.state_dict()
            if stop_early and self._should_stop_early(test_loss.history):
                if not quiet:
                    print("stopping early")
                break
        self.model.load_state_dict(best_model_state)
        self.model.eval()

    def _compute_distance(self, ps, ts):
        d = ([np.linalg.norm(p - t, ord=2) for p, t in zip(ps, ts)])
        return d

    @property
    def denormalized_test_set(self):
        return self.move_positions_back(self.test_set)

    @property
    def denormalized_train_set(self):
        return self.move_positions_back(self.train_set)

    @property
    def denormalized_full_set(self):
        return self.move_positions_back(self.full_set)

    def plot_history(self):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        train = self.train_loss.history
        test = self.test_loss.history
        ax.plot(range(len(train)), train, label="train")
        ax.plot(range(len(test)), test, label="test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train History")
        ax.legend()

    def _call_model_and_return_denormalized_prediction(self, sample):
        return self.move_positions_back(self.model(sample))

    def predict_position(self, sample):
        return self.move_positions_back(self.model(sample))

    def _get_prediction_and_target(self, ds):
        data = collect_prediction_target(self._call_model_and_return_denormalized_prediction, ds)
        data['distance'] = self._compute_distance(data['prediction'], data['target'])
        return data

    def get_prediction_target_distance_for_test(self):
        return self._get_prediction_and_target(self.test_set)

    def get_prediction_target_distance_for_train(self):
        return self._get_prediction_and_target(self.train_set)

    def get_prediction_target_distance(self):
        return self._get_prediction_and_target(self.full_set)

    def get_mean_and_var_distance_for_test(self):
        distances = self.get_prediction_target_distance_for_test()['distance']
        mean = sum(distances)/len(distances)
        var = sqrt(sum((d-mean)**2 for d in distances))/len(distances)
        return mean, var




