"""Synthetic medical dataset generator and partitioner for FedMed."""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure FedMed project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE,
    CLIENT_CLASS_DISTRIBUTIONS,
    DATASET_SIZE_PER_CLIENT,
    IMAGE_DIMS,
    RANDOM_SEED,
    TRAIN_SPLIT,
)


class SyntheticMedicalDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic 28x28 grayscale medical scans."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def _create_sample_image(class_label: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single 1x28x28 deterministic synthetic medical image.

    Class 0 represents a central dense lesion (Gaussian core pattern).
    Class 1 represents a peripheral annular rim / cortical lesion (ring pattern).
    """
    height, width = IMAGE_DIMS[1], IMAGE_DIMS[2]
    y, x = np.ogrid[-height / 2 : height / 2, -width / 2 : width / 2]
    radius = np.sqrt(x**2 + y**2)

    if class_label == 0:
        # Central Gaussian pattern
        sigma = 3.8 + rng.uniform(-0.3, 0.3)
        pattern = np.exp(-(radius**2) / (2.0 * sigma**2))
    else:
        # Peripheral ring pattern
        target_radius = 8.5 + rng.uniform(-0.5, 0.5)
        sigma = 2.2 + rng.uniform(-0.2, 0.2)
        pattern = np.exp(-((radius - target_radius) ** 2) / (2.0 * sigma**2))

    # Add realistic medical scanner background noise & subtle texture
    noise = rng.normal(loc=0.0, scale=0.08, size=(height, width))
    image = pattern + noise
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    # Shape: (1, 28, 28)
    return np.expand_dims(image, axis=0)


def generate_client_data(
    client_id: int,
    total_samples: int = DATASET_SIZE_PER_CLIENT,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic non-IID partitioned dataset for a specific client.

    Client 0 has 80% Class 0 (focal core) and 20% Class 1 (peripheral ring).
    Client 1 has 20% Class 0 (focal core) and 80% Class 1 (peripheral ring).
    """
    rng = np.random.default_rng(seed + client_id * 100)

    dist = CLIENT_CLASS_DISTRIBUTIONS.get(client_id, {0: 0.5, 1: 0.5})
    num_c0 = int(round(total_samples * dist[0]))
    num_c1 = total_samples - num_c0

    images = []
    labels = []

    # Generate Class 0 samples
    for _ in range(num_c0):
        images.append(_create_sample_image(class_label=0, rng=rng))
        labels.append(0)

    # Generate Class 1 samples
    for _ in range(num_c1):
        images.append(_create_sample_image(class_label=1, rng=rng))
        labels.append(1)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle samples deterministically
    shuffle_indices = rng.permutation(total_samples)
    images = images[shuffle_indices]
    labels = labels[shuffle_indices]

    return images, labels


def get_client_dataloaders(
    client_id: int,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    train_split: float = TRAIN_SPLIT,
    total_samples: int = DATASET_SIZE_PER_CLIENT,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for a client with an 80/20 train/test split.

    Returns:
        (train_loader, test_loader)
    """
    images, labels = generate_client_data(client_id, total_samples=total_samples, seed=seed)
    full_dataset = SyntheticMedicalDataset(images, labels)

    train_size = int(len(full_dataset) * train_split)
    test_size = len(full_dataset) - train_size

    # Deterministic split generator
    split_gen = torch.Generator().manual_seed(seed + client_id)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=split_gen
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + client_id),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
