"""Centralized configuration for FedMed federated learning setup."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Network Configuration
SERVER_ADDRESS: str = "127.0.0.1:8080"

# Federated Learning Configuration
NUM_CLIENTS: int = 3
NUM_ROUNDS: int = 3
LOCAL_EPOCHS: int = 2

# Training Hyperparameters
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
RANDOM_SEED: int = 42

# Data and Model Configuration
NUM_CLASSES: int = 2
IMAGE_DIMS: tuple = (1, 28, 28)  # (Channels, Height, Width)
DATASET_SIZE_PER_CLIENT: int = 200
TRAIN_SPLIT: float = 0.8

# Non-IID Partitioning Ratios:
# Client 0: 80% Class 0, 20% Class 1
# Client 1: 20% Class 0, 80% Class 1
CLIENT_CLASS_DISTRIBUTIONS = {
    0: {0: 0.80, 1: 0.20},
    1: {0: 0.20, 1: 0.80},
    2: {0: 0.50, 1: 0.50},
}

# Output Files
FINAL_MODEL_PATH = RESULTS_DIR / "final_model.pt"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"
