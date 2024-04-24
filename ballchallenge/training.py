from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


class LossMetric:
    def __init__(self):
        self._running_val = 0
        self._num_updates = 0
        self.history = []

    def update(self, val):
        self._running_val += val
        self._num_updates += 1

    @property
    def value(self):
        if self._num_updates == 0:
            return 500
        return self._running_val / self._num_updates

    def reset(self):
        self.history.append(self.value)
        self._running_val = 0
        self._num_updates = 0


@dataclass
class TrainStepResult:
    loss: float
    predictions: np.ndarray


def create_test_steps_factory(model: torch.nn.Module, test_data, device, loss_fn):
    def step() -> Iterator[TrainStepResult]:
        train_mode = model.training
        model.eval()
        with torch.no_grad():
            for batch in test_data:
                samples, labels = batch
                samples = samples.to(device)
                labels = labels.to(device)

                predictions = model(samples)
                loss = loss_fn(predictions, labels)
                yield TrainStepResult(loss.item(), predictions.detach().numpy())
        model.train(train_mode)
    return step


def create_train_steps_factory(model: torch.nn.Module, training_data, device, loss_fn, optimizer):
    def step() -> Iterator[TrainStepResult]:
        train_mode = model.training
        model.train()
        for batch in training_data:
            samples, labels = batch
            samples = samples.to(device)
            labels = labels.to(device)

            predictions = model(samples)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            yield TrainStepResult(loss.item(), predictions.detach().numpy())
        model.train(train_mode)
    return step
