"""Flower server implementation for FedMed federated learning."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure FedMed project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import torch

from src.config import (
    FINAL_MODEL_PATH,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RESULTS_DIR,
    SERVER_ADDRESS,
    TRAINING_HISTORY_PATH,
    METRICS_WEBSOCKET_HOST,
    METRICS_WEBSOCKET_PORT,
)
from src.config import MRI_IN_CHANNELS, MRI_OUT_CHANNELS, MRI_UNET_CHANNELS, MRI_UNET_STRIDES
from src.fedmed.models import UNet3DConfig, build_unet3d
from src.live_metrics import LiveMetricsPublisher
from src.model import get_parameters, set_parameters


def build_federated_model() -> torch.nn.Module:
    """Construct the exact U-Net topology used by every hospital client."""
    return build_unet3d(
        UNet3DConfig(
            in_channels=MRI_IN_CHANNELS,
            out_channels=MRI_OUT_CHANNELS,
            channels=MRI_UNET_CHANNELS,
            strides=MRI_UNET_STRIDES,
        )
    )


def evaluate_metrics_aggregation_fn(
    eval_metrics: List[Tuple[int, Metrics]]
) -> Metrics:
    """Compute sample-weighted average accuracy across all evaluated clients."""
    total_examples = sum([num_examples for num_examples, _ in eval_metrics])
    if total_examples == 0:
        return {"accuracy": 0.0}

    weighted_accuracy_sum = sum(
        [num_examples * float(m.get("accuracy", 0.0)) for num_examples, m in eval_metrics]
    )
    aggregated_accuracy = weighted_accuracy_sum / total_examples
    return {"accuracy": aggregated_accuracy}


class FedMedStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy tracking parameters, latencies, and saving history."""

    def __init__(self, *args, metrics_publisher: LiveMetricsPublisher | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latest_parameters: Optional[Parameters] = None
        self.history: List[Dict[str, Union[int, float]]] = []
        self.round_latencies: Dict[int, float] = {}
        self.metrics_publisher = metrics_publisher

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.latest_parameters = parameters

        # Aggregate reported client training latencies
        latencies = [
            float(res.metrics["latency_sec"])
            for _, res in results
            if res.metrics and "latency_sec" in res.metrics
        ]
        if latencies:
            self.round_latencies[server_round] = float(sum(latencies) / len(latencies))

        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        acc = metrics_aggregated.get("accuracy", 0.0) if metrics_aggregated else 0.0
        loss_val = float(loss_aggregated) if loss_aggregated is not None else 0.0
        avg_lat = self.round_latencies.get(server_round, 0.0)

        record = {
            "round": server_round,
            "loss": loss_val,
            "accuracy": float(acc),
            "latency_sec": round(avg_lat, 3),
        }
        self.history.append(record)
        if self.metrics_publisher is not None:
            self.metrics_publisher.publish_round(server_round, loss_val, float(acc), avg_lat)

        lat_str = f" | Avg Client Latency: {avg_lat:.2f}s" if avg_lat > 0 else ""
        print(
            f"\n>>> [Server Round {server_round} Aggregated Evaluation] "
            f"Loss: {loss_val:.5f} | Weighted Accuracy: {acc:.4%}{lat_str}\n",
            flush=True,
        )

        return loss_aggregated, metrics_aggregated


def fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Provide configuration parameters to clients during fit rounds."""
    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
    }


def eval_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Provide configuration parameters to clients during evaluation rounds."""
    return {
        "server_round": server_round,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMed Federated Learning Server")
    parser.add_argument(
        "--server-address",
        type=str,
        default=SERVER_ADDRESS,
        help=f"Server host and port (default: {SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--metrics-host",
        type=str,
        default=METRICS_WEBSOCKET_HOST,
        help=f"WebSocket host for live aggregate metrics (default: {METRICS_WEBSOCKET_HOST})",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=METRICS_WEBSOCKET_PORT,
        help=f"WebSocket port for live aggregate metrics (default: {METRICS_WEBSOCKET_PORT})",
    )
    parser.add_argument(
        "--disable-live-metrics",
        action="store_true",
        help="Disable the WebSocket metric stream.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=NUM_ROUNDS,
        help=f"Number of federated training rounds (default: {NUM_ROUNDS})",
    )
    args = parser.parse_args()

    print("==================================================", flush=True)
    print("  FedMed Federated Learning Server Initializing", flush=True)
    print(f"  Address: {args.server_address}", flush=True)
    print(f"  Rounds: {args.num_rounds}", flush=True)
    print(f"  Minimum Clients: {NUM_CLIENTS}", flush=True)
    print(
        f"  Live Metrics: {'disabled' if args.disable_live_metrics else f'ws://{args.metrics_host}:{args.metrics_port}'}",
        flush=True,
    )
    print("==================================================", flush=True)

    # Initialize global model
    initial_model = build_federated_model()
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))

    metrics_publisher = None
    if not args.disable_live_metrics:
        metrics_publisher = LiveMetricsPublisher(args.metrics_host, args.metrics_port)
        if metrics_publisher.start():
            print(f"[Server] Live metrics WebSocket listening on {metrics_publisher.url}", flush=True)
        else:
            print("[Server] Live metrics unavailable; continuing without WebSocket stream.", flush=True)
            metrics_publisher = None

    # Configure Strategy
    strategy = FedMedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=eval_config_fn,
        initial_parameters=initial_parameters,
        metrics_publisher=metrics_publisher,
    )

    # Start Flower Server
    try:
        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
        )
    except Exception as e:
        print(f"[Server] Error during federated session: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if metrics_publisher is not None:
            metrics_publisher.stop()

    print("\n==================================================", flush=True)
    print("  Federated Training Finished. Saving Artifacts...", flush=True)
    print("==================================================", flush=True)

    # 1. Save final global model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if strategy.latest_parameters is not None:
        final_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)
        final_model = build_federated_model()
        set_parameters(final_model, final_ndarrays)
        torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
        print(f"[Server] Saved final global model to: {FINAL_MODEL_PATH}", flush=True)
    else:
        # Fallback to initial if no rounds ran
        torch.save(initial_model.state_dict(), FINAL_MODEL_PATH)
        print(f"[Server] Saved initial model to: {FINAL_MODEL_PATH}", flush=True)

    # 2. Save training history CSV
    fieldnames = ["round", "loss", "accuracy", "latency_sec"]
    with open(TRAINING_HISTORY_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in strategy.history:
            writer.writerow(row)
    print(f"[Server] Saved training history to: {TRAINING_HISTORY_PATH}", flush=True)

    # Print summary table
    print("\n--- FedMed Round Summary ---", flush=True)
    print(f"{'Round':<8}{'Aggregated Loss':<18}{'Aggregated Accuracy':<22}{'Avg Latency':<14}", flush=True)
    print("-" * 62, flush=True)
    for row in strategy.history:
        lat = f"{row.get('latency_sec', 0.0):.2f}s"
        print(
            f"{row['round']:<8}{row['loss']:<18.5f}{row['accuracy'] * 100:>17.2f}%   {lat:<14}",
            flush=True,
        )
    print("-" * 62, flush=True)


if __name__ == "__main__":
    main()
