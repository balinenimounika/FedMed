"""Flower client implementation for FedMed federated learning."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure FedMed project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl
import torch

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    RANDOM_SEED,
    SERVER_ADDRESS,
)
from src.dataset import get_client_dataloaders
from src.model import MedicalCNN, get_parameters, set_parameters, test, train


class FedMedClient(fl.client.NumPyClient):
    """NumPyClient that trains and evaluates MedicalCNN on local medical partitions."""

    def __init__(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = MedicalCNN().to(self.device)
        print(
            f"[Client {self.client_id}] Initialized with {len(self.train_loader.dataset)} "
            f"training samples and {len(self.test_loader.dataset)} test samples on {self.device}.",
            flush=True,
        )

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List:
        return get_parameters(self.model)

    def fit(
        self, parameters: List, config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List, int, Dict]:
        # Update local model with global parameters
        set_parameters(self.model, parameters)

        server_round = config.get("server_round", "?")
        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))

        start_time = time.perf_counter()
        print(
            f"\n[Client {self.client_id}] Starting Local Training (Round {server_round}, {epochs} epochs)...",
            flush=True,
        )
        loss, accuracy = train(
            self.model,
            self.train_loader,
            epochs=epochs,
            device=self.device,
            learning_rate=LEARNING_RATE,
            client_id=self.client_id,
            verbose=True,
        )
        duration_sec = time.perf_counter() - start_time
        print(
            f"[Client {self.client_id}] Completed Training ({duration_sec:.2f}s) - Final Loss: {loss:.5f}, Final Accuracy: {accuracy * 100:.2f}%",
            flush=True,
        )

        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "latency_sec": round(float(duration_sec), 3),
            },
        )

    def evaluate(
        self, parameters: List, config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        server_round = config.get("server_round", "?")

        loss, accuracy = test(self.model, self.test_loader, device=self.device)
        print(
            f"[Client {self.client_id}] Evaluation (Round {server_round}) - Test Loss: {loss:.5f}, Test Accuracy: {accuracy * 100:.2f}%",
            flush=True,
        )

        return (
            float(loss),
            len(self.test_loader.dataset),
            {"accuracy": float(accuracy), "loss": float(loss)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMed Flower Client")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        choices=[0, 1],
        help="Client identifier (0 or 1)",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=SERVER_ADDRESS,
        help=f"FedMed server address (default: {SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for DataLoader (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"==================================================", flush=True)
    print(f"  FedMed Client {args.client_id} Starting", flush=True)
    print(f"  Target Server: {args.server_address}", flush=True)
    print(f"==================================================", flush=True)

    train_loader, test_loader = get_client_dataloaders(
        client_id=args.client_id,
        batch_size=args.batch_size,
        seed=RANDOM_SEED,
    )

    client = FedMedClient(
        client_id=args.client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )

    try:
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
    except Exception as e:
        print(f"[Client {args.client_id}] Communication error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[Client {args.client_id}] Federated learning session ended successfully.", flush=True)


if __name__ == "__main__":
    main()
