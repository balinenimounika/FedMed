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
# Client 0 (Hospital A): 80% Class 0, 20% Class 1 (Focal lesion center)
# Client 1 (Hospital B): 20% Class 0, 80% Class 1 (Peripheral rim center)
# Client 2 (Hospital C): 50% Class 0, 50% Class 1 (General community center)
CLIENT_CLASS_DISTRIBUTIONS = {
    0: {0: 0.80, 1: 0.20},
    1: {0: 0.20, 1: 0.80},
    2: {0: 0.50, 1: 0.50},
}

# Output Files
FINAL_MODEL_PATH = RESULTS_DIR / "final_model.pt"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"

# Week 6: 3D MRI segmentation integration. The compact volume and channel
# configuration makes a hospital-side smoke test feasible on CPU while keeping
# a compatible MONAI U-Net topology for Flower parameter exchange.
MRI_VOLUME_DIMS: tuple = (16, 16, 16)  # (Depth, Height, Width)
MRI_IN_CHANNELS: int = 1
MRI_OUT_CHANNELS: int = 2  # background, lesion
MRI_UNET_CHANNELS: tuple = (4, 8, 16, 32)
MRI_UNET_STRIDES: tuple = (2, 2, 2)
MRI_DATASET_SIZE_PER_CLIENT: int = 8
MRI_BATCH_SIZE: int = 1
MRI_TRAINING_RESULTS_PATH = RESULTS_DIR / "week6_local_training_results.json"

# Live metric stream exposed by the central aggregator. The endpoint publishes
# aggregate round metrics only; no patient data or model parameters are sent.
METRICS_WEBSOCKET_HOST: str = "127.0.0.1"
METRICS_WEBSOCKET_PORT: int = 8765
