from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from .train_history import TrainHistory


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

    loss_fn = torch.nn.CrossEntropyLoss()
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
