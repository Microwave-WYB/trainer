from abc import ABC
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


@dataclass
class Metric:
    compute: Callable[[torch.Tensor, torch.Tensor], float]
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.compute.__name__

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return self.compute(outputs, targets)


@dataclass
class EpochResult:
    epoch: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]


class TrainerPlugin(ABC):
    def on_train_begin(self, trainer: "Trainer") -> None: ...

    def on_train_end(self, trainer: "Trainer") -> None: ...

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None: ...

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None: ...

    def on_batch_end(self, trainer: "Trainer", batch: int, metrics: dict[str, float]) -> None: ...


class Trainer:
    """Easy-to-use training loop for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: optim.Optimizer,  # type: ignore
        metrics: list[Metric],
        total_epochs: int,
        plugins: Optional[list[TrainerPlugin]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.max_epochs = total_epochs
        self.plugins = plugins or []
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.current_epoch = 0
        self._stop = False
        self.history: list[EpochResult] = []

    def __iter__(self) -> Iterator[EpochResult]:
        """Iterate over the epoch results indefinitely."""
        for callback in self.plugins:
            callback.on_train_begin(self)

        while not self._stop and self.current_epoch < self.max_epochs:
            self.current_epoch += 1
            for callback in self.plugins:
                callback.on_epoch_begin(self, self.current_epoch)

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            epoch_result = EpochResult(
                epoch=self.current_epoch, train_metrics=train_metrics, val_metrics=val_metrics
            )

            for callback in self.plugins:
                callback.on_epoch_end(self, self.current_epoch, {**train_metrics, **val_metrics})

            self.history.append(epoch_result)
            yield epoch_result

            if self._stop:
                break

        for callback in self.plugins:
            callback.on_train_end(self)

    def train_epoch(self) -> dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        metrics = {metric.name: 0.0 for metric in self.metrics}
        metrics["loss"] = 0.0

        for batch, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            metrics["loss"] += loss.item()
            for metric in self.metrics:
                metrics[metric.name] += metric(outputs, targets)

            batch_metrics = {k: v / (batch + 1) for k, v in metrics.items()}
            for callback in self.plugins:
                callback.on_batch_end(self, batch, batch_metrics)

        return {k: v / len(self.train_loader) for k, v in metrics.items()}

    def validate(self) -> dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics = {metric.name: 0.0 for metric in self.metrics}
        metrics["loss"] = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                metrics["loss"] += loss.item()
                for metric in self.metrics:
                    metrics[metric.name] += metric(outputs, targets)

        return {k: v / len(self.val_loader) for k, v in metrics.items()}

    def load_plugin(self, *plugins: TrainerPlugin) -> None:
        """Add plugins to the trainer."""
        self.plugins.extend(plugins)

    def stop(self):
        """Set the stop flag to True to end training."""
        self._stop = True

    def resume(self):
        """Set the stop flag to False to resume training."""
        self._stop = False
