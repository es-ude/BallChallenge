from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .train_history import TrainHistory


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


def run_training_for_position(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_test: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: Any,
) -> TrainHistory:
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model.to(device)

    loss_fn = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainHistory()
    print("start training")
    best_loss = -1
    best_model_state = None
    unsuccessfull_tries_to_improve_test_loss = 0
    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0
        num_samples = 0

        for samples, labels in dl_train:
            samples = samples.to(device)
            labels = labels.to(device)

            predictions = model(samples)
            loss = loss_fn(predictions, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            num_samples += len(samples)

        train_loss = running_loss / len(dl_train)

        model.eval()

        running_loss = 0
        num_samples = 0

        with torch.no_grad():
            for samples, labels in dl_test:
                samples = samples.to(device)
                labels = labels.to(device)

                predictions = model(samples)
                loss = loss_fn(predictions, labels)

                running_loss += loss.item()
                num_samples += len(samples)

        test_loss = running_loss / len(dl_test)
        if best_loss < 0 or test_loss < best_loss:
            best_model_state = model.state_dict()
            unsuccessfull_tries_to_improve_test_loss = 0
        else:
            unsuccessfull_tries_to_improve_test_loss += 1
            if unsuccessfull_tries_to_improve_test_loss > 30:
                return history, best_model_state
        history.log("epoch", epoch, epoch)
        history.log("loss", train_loss, test_loss)

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss: {train_loss:.04f} "
            f"test_loss: {test_loss:.04f} "
        )

    return history, best_model_state

def _correct_predicted(
    predicted_labels: torch.Tensor, target_labels: torch.Tensor
) -> int:
    def extract_class_indices(labels: torch.Tensor) -> torch.Tensor:
        return torch.tensor([label.argmax() for label in labels])

    predicted_indices = extract_class_indices(predicted_labels)
    target_indices = extract_class_indices(target_labels)

    return int((predicted_indices == target_indices).sum().item())


def run_training(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_test: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: Any,
) -> TrainHistory:
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model.to(device)

    def loss_fn(x):
        smax = torch.nn.functional.softmax
        cle = torch.nn.functional.cross_entropy
        return cle(smax(x, dim=1))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainHistory()

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0

        for samples, labels in dl_train:
            samples = samples.to(device)
            labels = labels.to(device)

            model.zero_grad()

            predictions = model(samples)
            loss = loss_fn(predictions, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(dl_train)

        model.eval()

        running_loss = 0

        with torch.no_grad():
            for samples, labels in dl_test:
                samples = samples.to(device)
                labels = labels.to(device)

                predictions = model(samples)
                loss = loss_fn(predictions, labels)

                running_loss += loss.item()

        test_loss = running_loss / len(dl_test)

        history.log("epoch", epoch, epoch)
        history.log("loss", train_loss, test_loss)

        print(
            f"[epoch {epoch}/{epochs}] train_loss: {train_loss:.04f} ; test_loss: {test_loss:.04f}"
        )

    return history
