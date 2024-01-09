class TrainHistory:
    def __init__(self) -> None:
        self._metrics: dict[str, dict[str, list[float]]] = dict(
            train=dict(), test=dict()
        )

    @property
    def train(self) -> dict[str, list[float]]:
        return self._metrics["train"]

    @property
    def test(self) -> dict[str, list[float]]:
        return self._metrics["test"]

    def log(self, metric: str, train_value: float, test_value: float) -> None:
        if metric not in self.train:
            self._metrics["train"][metric] = []
            self._metrics["test"][metric] = []

        self._metrics["train"][metric].append(train_value)
        self._metrics["test"][metric].append(test_value)
