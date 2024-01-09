from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader

from .train_history import TrainHistory


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
    learning_rate: int,
    device: Any,
) -> TrainHistory:
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model.to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainHistory()

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0
        correct_predicted = 0
        num_samples = 0

        for samples, labels in dl_train:
            samples = samples.to(device)
            labels = labels.to(device)

            model.zero_grad()

            predictions = model(samples)
            loss = loss_fn(predictions, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predicted += _correct_predicted(predictions, labels)
            num_samples += len(samples)

        train_loss = running_loss / len(dl_train)
        train_accuracy = correct_predicted / num_samples

        model.eval()

        running_loss = 0
        correct_predicted = 0
        num_samples = 0

        with torch.no_grad():
            for samples, labels in dl_test:
                samples = samples.to(device)
                labels = labels.to(device)

                predictions = model(samples)
                loss = loss_fn(predictions, labels)

                running_loss += loss.item()
                correct_predicted += _correct_predicted(predictions, labels)
                num_samples += len(samples)

        test_loss = running_loss / len(dl_test)
        test_accuracy = correct_predicted / num_samples

        history.log("epoch", epoch, epoch)
        history.log("loss", train_loss, test_loss)
        history.log("accuracy", train_accuracy, test_accuracy)

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss: {train_loss:.04f} ; train_accuracy: {train_accuracy:.04f} ; "
            f"test_loss: {test_loss:.04f} ; test_accuracy: {test_accuracy:.04f}"
        )

    return history
