from typing import override

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from trainer.core import Metric, Trainer
from trainer.plugins import EarlyStopping, ProgressBar
from trainer.plugins.checkpoint import Checkpoint, load_latest_checkpoint_to_trainer


# Define metric
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == targets).sum().item() / targets.size(0)


# Define models
class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)  # Changed from Dropout2d to Dropout
        self.dropout2 = nn.Dropout(0.5)  # Changed from Dropout2d to Dropout
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)


def train_mnist(model: nn.Module, model_name: str) -> None:
    batch_size = 64
    total_epochs = 3

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=nn.NLLLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),  # type: ignore
        metrics=[Metric(compute=accuracy)],
        total_epochs=total_epochs,
        plugins=[
            ProgressBar(),
            EarlyStopping("loss", 3),
            Checkpoint(f"./checkpoints/{model_name}"),
        ],
    )

    try:
        load_latest_checkpoint_to_trainer(f"./checkpoints/{model_name}", trainer)
    except FileNotFoundError:
        print(f"No checkpoints found for {model_name}. Starting from scratch.")

    for epoch_result in trainer:
        # Your custom logic between epochs here
        pass

    print(f"Final {model_name} accuracy: {trainer.history[-1].val_metrics['accuracy']:.4f}")


# Run training for both models
if __name__ == "__main__":
    cnn_model = SimpleCNN()
    mlp_model = SimpleMLP()

    train_mnist(cnn_model, "CNN")
    train_mnist(mlp_model, "MLP")
