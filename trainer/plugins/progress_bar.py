from typing import Any

from tqdm import tqdm

from trainer.core import Trainer, TrainerPlugin
from trainer.utils import unwrap


class ProgressBar(TrainerPlugin):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, trainer: Trainer) -> None:
        total_batches = trainer.max_epochs * len(trainer.train_loader)
        self.pbar = tqdm(total=total_batches, desc=f"Epoch {trainer.current_epoch}", mininterval=1)
        self.pbar.update(trainer.current_epoch * len(trainer.train_loader))

    def on_train_end(self, trainer: Trainer) -> None:
        if self.pbar:
            self.pbar.close()

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        unwrap(self.pbar).desc = f"Epoch {epoch}"

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        epoch_metrics = " ".join([f"val_{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch}: {epoch_metrics}")

    def on_batch_end(self, trainer: Trainer, batch: int, metrics: dict[str, float]) -> None:
        unwrap(self.pbar).update(1)
        info: dict[str, Any] = {k: f"{v:.4f}" for k, v in metrics.items()}
        unwrap(self.pbar).set_postfix(info)
