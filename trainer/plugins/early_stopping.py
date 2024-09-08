from typing import Literal, Optional

from trainer.core import Trainer, TrainerPlugin


class EarlyStopping(TrainerPlugin):
    """Early stopping plugin to stop training when the monitored metric stops improving."""

    def __init__(
        self,
        monitor: str,
        patience: int,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        baseline: Optional[float] = None,
    ):
        """
        Args:
            monitor (str): The metric name to monitor. For example, "loss".
            patience (int): Number of epochs to wait before stopping.
            min_delta (float, optional): minimum improvement. Defaults to 0.
            mode (Literal[&quot;min&quot;, &quot;max&quot;], optional): min or max. Defaults to &quot;min&quot;.
            baseline (Optional[float], optional): The baseline value to compare against. Defaults to None.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.best_score: Optional[float] = None
        self.counter = 0
        self.stopped_epoch = 0

    def on_train_begin(self, trainer: Trainer) -> None:
        self.best_score = None
        self.counter = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            raise ValueError(f"Monitored metric '{self.monitor}' is not available in metrics.")

        match self.best_score:
            case None:
                self.best_score = current
            case float() if self.mode == "min":
                if current < self.best_score - self.min_delta:
                    self.best_score = current
                    self.counter = 0
                else:
                    self.counter += 1
            case float() if self.mode == "max":
                if current > self.best_score + self.min_delta:
                    self.best_score = current
                    self.counter = 0
                else:
                    self.counter += 1
            case _:
                raise ValueError(f"Mode '{self.mode}' is not supported. Use 'min' or 'max'.")

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            trainer.stop()

    def on_train_end(self, trainer: Trainer) -> None:
        if self.stopped_epoch > 0:
            print(f"Early stopping occurred at epoch {self.stopped_epoch}")
