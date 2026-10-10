
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
from src.encryption import (
    CKKSConfig,
    create_ckks_context,
    encrypt_vector_chunks,
    ensure_consortium_keys,
    extract_and_flatten_state_dict,
    homomorphic_fedavg,
    load_context_from_file,
    save_context_to_file,
)
from src.fedmed.models import UNet3DConfig, build_unet3d
from src.live_metrics import LiveMetricsPublisher
from src.model import get_parameters, set_parameters
import tenseal as ts
import time


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


class FedMedCKKSStrategy(fl.server.strategy.FedAvg):
    """
    Homomorphic Encryption Strategy for FedMed.
    Executes sample-weighted federated averaging directly over TenSEAL CKKS
    ciphertext vectors without access to institutional private keys.
    """

    def __init__(
        self,
        public_context: ts.Context,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.public_context = public_context

        # Enforce cryptographic zero-knowledge invariant
        if self.public_context.has_secret_key():
            raise ValueError("Cryptographic Fault: Server context MUST NOT contain the private secret key!")

        self.latest_encrypted_parameters: Optional[Parameters] = None
        self.history: List[Dict[str, Union[int, float]]] = []
        self.round_latencies: Dict[int, float] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        if not results:
            return None, {}

        # 1. Collect client ciphertexts and institutional sample weights
        client_encrypted_chunks: List[List[bytes]] = []
        client_weights: List[float] = []

        for _, fit_res in results:
            client_encrypted_chunks.append(fit_res.parameters.tensors)
            client_weights.append(float(fit_res.num_examples))

        # 2. Homomorphic FedAvg Aggregation directly over Ciphertext Space
        t_agg_start = time.perf_counter()
        aggregated_chunks = homomorphic_fedavg(
            self.public_context,
            client_encrypted_chunks,
            client_weights,
        )
        agg_time = time.perf_counter() - t_agg_start

        # 3. Encapsulate aggregated ciphertext into Flower Parameters
        aggregated_parameters = Parameters(
            tensors=aggregated_chunks,
            tensor_type="tenseal_ckks",
        )
        self.latest_encrypted_parameters = aggregated_parameters

        # Aggregate reported client training latencies
        latencies = [
            float(res.metrics["latency_sec"])
            for _, res in results
            if res.metrics and "latency_sec" in res.metrics
        ]
        if latencies:
            self.round_latencies[server_round] = float(sum(latencies) / len(latencies))

        total_bytes = sum(len(c) for c in aggregated_chunks)
        print(
            f"\n>>> [Server Round {server_round} Secure HE-FedAvg] "
            f"Aggregated {len(results)} encrypted institutional updates ({len(aggregated_chunks)} chunks, "
            f"{total_bytes / (1024*1024):.2f} MB) in {agg_time:.4f}s directly over ciphertext (Zero Plaintext Exposure)\n",
            flush=True,
        )

        metrics_aggregated: Dict[str, fl.common.Scalar] = {
            "he_aggregation_sec": round(agg_time, 4),
            "num_aggregated_clients": len(results),
        }
        return aggregated_parameters, metrics_aggregated

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
    parser.add_argument(
        "--encrypted",
        action="store_true",
        help="Enable TenSEAL CKKS homomorphic aggregation strategy",
    )
    parser.add_argument(
        "--context-path",
        type=str,
        default=None,
        help="Path to public CKKS server context file (.seal)",
    )
    args = parser.parse_args()

    mode_label = "ENCRYPTED (TenSEAL CKKS)" if args.encrypted else "STANDARD (Plaintext)"
    print("==================================================", flush=True)
    print(f"   FedMed Federated Learning Server Initializing [{mode_label}]", flush=True)
    print(f"   Address: {args.server_address}", flush=True)
    print(f"   Rounds: {args.num_rounds}", flush=True)
    print(f"   Minimum Clients: {NUM_CLIENTS}", flush=True)
    print(
        f"   Live Metrics: {'disabled' if args.disable_live_metrics else f'ws://{args.metrics_host}:{args.metrics_port}'}",
        flush=True,
    )
    print("==================================================", flush=True)

    # Initialize global model
    initial_model = build_federated_model()
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))

    # Initialize live metrics publisher if enabled
    metrics_publisher = None
    if not args.disable_live_metrics:
        try:
            metrics_publisher = LiveMetricsPublisher(
                host=args.metrics_host,
                port=args.metrics_port,
            )
            metrics_publisher.start()
        except Exception as e:
            print(f"[Server] Warning: Could not start LiveMetricsPublisher: {e}", flush=True)

    # Configure Strategy
    if args.encrypted:
        ctx_path = Path(args.context_path) if args.context_path else Path("keys/ckks_public.seal")
        if not ctx_path.is_file():
            print(f"[Server] Error: Public CKKS context not found at {ctx_path}. Run verify_encryption.py first.", file=sys.stderr)
            sys.exit(1)
        public_context = load_context_from_file(ctx_path)
        strategy = FedMedCKKSStrategy(
            public_context=public_context,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=eval_config_fn,
            initial_parameters=initial_parameters,
        )
    else:
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
    print("   Federated Training Finished. Saving Artifacts...", flush=True)
    print("==================================================", flush=True)

    # 1. Save final global model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.encrypted:
        if hasattr(strategy, "latest_encrypted_parameters") and strategy.latest_encrypted_parameters is not None:
            enc_model_path = RESULTS_DIR / "final_model_encrypted.seal"
            # Save raw encrypted chunks
            import pickle
            with open(enc_model_path, "wb") as f:
                pickle.dump(strategy.latest_encrypted_parameters.tensors, f)
            print(f"[Server] Saved final global encrypted model to: {enc_model_path}", flush=True)
            print("        Notice: Decryption strictly reserved for client institutions holding secret key.", flush=True)
    else:
        if hasattr(strategy, "latest_parameters") and strategy.latest_parameters is not None:
            final_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)
            final_model = build_federated_model()
            set_parameters(final_model, final_ndarrays)
            torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
            print(f"[Server] Saved final global model to: {FINAL_MODEL_PATH}", flush=True)
        else:
            torch.save(initial_model.state_dict(), FINAL_MODEL_PATH)
            print(f"[Server] Saved initial model to: {FINAL_MODEL_PATH}", flush=True)

    # 2. Save training history to CSV
    if hasattr(strategy, "history") and strategy.history:   
        with open(TRAINING_HISTORY_PATH, mode="w", newline="") as csv_file:
            fieldnames = ["round", "loss", "accuracy", "latency_sec"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in strategy.history:
                writer.writerow(record)
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