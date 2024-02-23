import itertools
import operator
from functools import partial, reduce
from math import sqrt
from pathlib import Path
from typing import cast, Optional, Sequence, Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


def bnormed_conv(in_channels, out_channels, kernel_size):
    return nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                         nn.BatchNorm1d(num_features=out_channels),
                         nn.Sigmoid())


def create_position_model(input_shape: tuple[int, int], kernel_sizes: Sequence[int] = (26, 26, 26), out_channels: Sequence[int]=(3, 3, 3), lin_act: Callable[[], nn.Module] = nn.Hardsigmoid, conv=bnormed_conv, linear_out_features: Sequence[int]=(2,)) -> nn.Module:
    in_length = input_shape[1]
    def flatten_linear(in_channels, out_features, kernel_sizes):
        in_features = in_length
        for k in kernel_sizes:
            in_features -= (k - 1)
        in_features = in_channels * in_features
        return nn.Sequential(nn.Flatten(), nn.Linear(in_features=in_features, out_features=out_features))

    def _conv(in_channels, out_channels, kernel_size):
        return conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    convs = []
    convs.append(_conv(in_channels=input_shape[0], out_channels=out_channels[0], kernel_size=kernel_sizes[0]))
    for k, oc, ic in zip(kernel_sizes[1:], out_channels[1:], out_channels[:-1]):
        convs.append(_conv(in_channels=ic, out_channels=oc, kernel_size=k))
    total_num_linears = len(linear_out_features)
    linears = [flatten_linear(in_channels=out_channels[-1], out_features=linear_out_features[0], kernel_sizes=kernel_sizes)]
    if total_num_linears > 1:
        linears.extend([nn.BatchNorm1d(linear_out_features[0]), lin_act()])
    for id, in_features, out_features in zip(range(2, total_num_linears+1), linear_out_features[:-1], linear_out_features[1:]):
        linears.append(nn.Linear(in_features=in_features, out_features=out_features))
        if id < total_num_linears:
            linears.extend([nn.BatchNorm1d(out_features), lin_act()])
    all_layers = tuple(chain(convs, linears))
    return nn.Sequential(
        *all_layers
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
    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
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


def bnormed_conv(in_channels, out_channels, kernel_size, act=nn.Sigmoid):
        return nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
                             act()
            )


def run_grid_search(experiment, search_space):

    total_number_of_configurations = reduce(operator.mul, (len(subspace) for subspace in search_space.values()))
    print("total number of configurations ", total_number_of_configurations)

    def expand_search_space(space):
        return [dict(zip(space.keys(), combination)) for combination in itertools.product(*tuple(space.values()))]

    tried_configs_and_mean = []
    best_mean = 2
    best_model = None
    trials = 2
    best_trial_id = 0
    for id, config in enumerate(expand_search_space(search_space)):
        for trial in range(trials):
            print(f"trial {2*id+trial} of {total_number_of_configurations*trials}")
            experiment.model = create_position_model(
                input_shape=experiment.input_shape,
                kernel_sizes=config['conv_setup']['kernel_sizes'],
                out_channels=config['conv_setup']['out_channels'],
                lin_act=config['lin_act'],
                linear_out_features=config['lin_out_features'],
                conv=partial(bnormed_conv, act=config['conv_act'])
            )
            experiment.train(900, quiet=True)
            mean, var = experiment.get_mean_and_var_distance_for_test()
            print(mean)
            print(config)

            if mean < best_mean:
                print("new best mean distance (meters) ", mean, "best var ", var)
                print(experiment.model)
                best_mean = mean
                best_model = experiment.model
                best_trial_id = 2*id+trial
            tried_configs_and_mean.append((config, mean, var))
    print("done.")
    print("best model ", best_model)
    print("best mean ", best_mean)
    return best_model, best_trial_id

