"""Lightweight PyTorch CNN model and helper routines for FedMed."""

from collections import OrderedDict
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


class MedicalCNN(nn.Module):
    """2-layer Conv2D CNN with 2 Linear layers for binary medical image classification."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Input: (batch_size, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (batch_size, 16, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (batch_size, 32, 7, 7)

        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)  # Raw logits for CrossEntropyLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load model parameters from a list of NumPy ndarrays into the PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def parameter_l1_norm(model: nn.Module) -> float:
    """Stable scalar fingerprint used to confirm local training changed weights."""
    return float(sum(parameter.detach().abs().sum().item() for parameter in model.parameters()))


def _is_segmentation_batch(outputs: torch.Tensor, labels: torch.Tensor) -> bool:
    return outputs.ndim == 5 and labels.ndim == 4


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
    learning_rate: float = 0.001,
    client_id: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Train the model for a given number of epochs on the specified device.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.to(device)
    model.train()
    criterion: nn.Module = DiceCELoss(to_onehot_y=True, softmax=True) if getattr(model, "spatial_dims", None) == 3 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_loss = 0.0
    epoch_acc = 0.0

    for epoch_idx in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            segmentation = _is_segmentation_batch(outputs, labels)
            loss = criterion(outputs, labels.unsqueeze(1) if segmentation else labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.numel() if segmentation else labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        if verbose:
            prefix = f"[Client {client_id}] " if client_id is not None else ""
            print(
                f"  {prefix}Local Epoch {epoch_idx}/{epochs} - Loss: {epoch_loss:.5f}, Acc: {epoch_acc * 100:.2f}%",
                flush=True,
            )

    return epoch_loss, epoch_acc


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Evaluate model on test dataset.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.to(device)
    model.eval()
    criterion: nn.Module = DiceCELoss(to_onehot_y=True, softmax=True) if getattr(model, "spatial_dims", None) == 3 else nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            segmentation = _is_segmentation_batch(outputs, labels)
            loss = criterion(outputs, labels.unsqueeze(1) if segmentation else labels)

            running_loss += loss.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.numel() if segmentation else labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return loss, accuracy
