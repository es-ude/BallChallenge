from matplotlib import pyplot as plt


def plot_training_history(history) -> None:
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.plot(history.train["epoch"], history.train["loss"], label="train")
    ax.plot(history.test["epoch"], history.test["loss"], label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train History")
    ax.legend()
