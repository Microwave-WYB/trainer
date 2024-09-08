from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

from trainer.core import EpochResult, Trainer, TrainerPlugin
from trainer.utils import unwrap


@dataclass
class CheckpointData:
    epoch: int
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]
    metrics: dict[str, float]
    history: list[EpochResult]


class Checkpoint(TrainerPlugin):
    """Checkpoint plugin to save the most current model and the best model."""

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "loss",
        mode: Literal["min", "max"] = "min",
        save_best_only: bool = False,
        save_frequency: int = 1,
        filename_prefix: str = "checkpoint",
    ):
        """
        Args:
            save_dir (Union[str, Path]): Directory to save the checkpoints.
            monitor (str): The metric name to monitor for the best model. For example, "val_loss".
            mode (Literal["min", "max"], optional): Whether the monitored metric should be minimized or maximized. Defaults to "min".
            save_best_only (bool, optional): If True, only saves the best model. Defaults to False.
            save_frequency (int, optional): How often to save the current model (in epochs). Defaults to 1.
            filename_prefix (str, optional): Prefix for the saved model filenames. Defaults to "checkpoint".
        """
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.filename_prefix = filename_prefix
        self.best_score: Optional[float] = None

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            raise ValueError(
                f"Monitor '{self.monitor}' not found in metrics. Available metrics are {list(metrics.keys())}."
            )
        current_score = unwrap(metrics.get(self.monitor))

        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and current_score < self.best_score:
            is_best = True
        elif self.mode == "max" and current_score > self.best_score:
            is_best = True

        if is_best:
            self.best_score = current_score
            self._save_checkpoint(trainer, epoch, metrics, is_best=True)

        if not self.save_best_only and epoch % self.save_frequency == 0:
            self._save_checkpoint(trainer, epoch, metrics, is_best=False)

    def _save_checkpoint(
        self, trainer: Trainer, epoch: int, metrics: dict[str, float], is_best: bool
    ) -> None:
        checkpoint = CheckpointData(
            epoch=epoch,
            model_state_dict=trainer.model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            metrics=metrics,
            history=trainer.history,
        )
        if is_best:
            filepath = (self.save_dir / self.filename_prefix).with_suffix("_best.pt")
            torch.save(checkpoint, filepath)
            print(f"Saved checkpoint to {filepath}")

        filepath = (self.save_dir / self.filename_prefix).with_suffix(f"_epoch_{epoch}.pt")
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")


def load_checkpoint_to_trainer(
    filepath: str | Path,
    trainer: Trainer,
):
    filepath = Path(filepath)
    checkpoint = torch.load(filepath)
    trainer.model.load_state_dict(checkpoint.model_state_dict)
    trainer.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    trainer.current_epoch = checkpoint.epoch
    trainer.history = checkpoint.history
    print(f"Loaded checkpoint from {filepath}")


def load_best_checkpoint_to_trainer(
    checkpoint_dir: str | Path,
    trainer: Trainer,
):
    filepath = Path(checkpoint_dir) / "checkpoint_best.pt"
    load_checkpoint_to_trainer(filepath, trainer)


def load_latest_checkpoint_to_trainer(
    checkpoint_dir: str | Path,
    trainer: Trainer,
) -> None:
    checkpoint_names = list(Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    if len(checkpoint_names) == 0:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_names, key=lambda p: p.stat().st_mtime)
    load_checkpoint_to_trainer(latest_checkpoint, trainer)


def load_checkpoint_to_model(
    filepath: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,  # type: ignore
) -> None:
    filepath = Path(filepath)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {filepath}")


def load_best_checkpoint_to_model(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,  # type: ignore
) -> None:
    filepath = Path(checkpoint_dir) / "checkpoint_best.pt"
    load_checkpoint_to_model(filepath, model, optimizer)


def load_latest_checkpoint_to_model(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,  # type: ignore
) -> None:
    checkpoint_names = list(Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    if len(checkpoint_names) == 0:
        raise FileNotFoundError("No checkpoint files found in the checkpoint directory.")
    latest_checkpoint = max(checkpoint_names, key=lambda p: p.stat().st_mtime)
    load_checkpoint_to_model(latest_checkpoint, model, optimizer)
