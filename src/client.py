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
    MRI_BATCH_SIZE,
    MRI_IN_CHANNELS,
    MRI_OUT_CHANNELS,
    MRI_UNET_CHANNELS,
    MRI_UNET_STRIDES,
)
from src.dataset import get_client_mri_dataloaders
from src.encryption import (
    CKKSConfig,
    create_ckks_context,
    decrypt_vector_chunks,
    encrypt_vector_chunks,
    ensure_consortium_keys,
    extract_and_flatten_state_dict,
    load_context_from_file,
    reconstruct_state_dict,
)
from src.fedmed.models import UNet3DConfig, build_unet3d
from src.model import get_parameters, parameter_l1_norm, set_parameters, test, train
import tenseal as ts


class FedMedClient(fl.client.NumPyClient):
    """Flower client that trains the shared MONAI 3D U-Net on local MRI data."""

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
        self.model = build_unet3d(
            UNet3DConfig(
                in_channels=MRI_IN_CHANNELS,
                out_channels=MRI_OUT_CHANNELS,
                channels=MRI_UNET_CHANNELS,
                strides=MRI_UNET_STRIDES,
            )
        ).to(self.device)
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
        weight_l1_before = parameter_l1_norm(self.model)
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
        weight_l1_after = parameter_l1_norm(self.model)
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
                "weight_l1_before": weight_l1_before,
                "weight_l1_after": weight_l1_after,
                "weight_l1_change": weight_l1_after - weight_l1_before,
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


class EncryptedFedMedClient(fl.client.Client):
    """
    Enterprise-grade Flower client that protects institutional patient privacy
    by encrypting local PyTorch model updates via TenSEAL CKKS homomorphic encryption.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        ckks_context: ts.Context,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.ckks_context = ckks_context

        # Verify client holds secret key
        if not self.ckks_context.is_private():
            raise ValueError(f"[Client {self.client_id}] Private CKKS context required for decryption!")

        self.model = build_unet3d(
            UNet3DConfig(
                in_channels=MRI_IN_CHANNELS,
                out_channels=MRI_OUT_CHANNELS,
                channels=MRI_UNET_CHANNELS,
                strides=MRI_UNET_STRIDES,
            )
        ).to(self.device)

        # Precompute tensor structure metadata for reconstruction
        initial_flat, self.metadata = extract_and_flatten_state_dict(self.model.state_dict())
        self.total_params = len(initial_flat)
        self.slot_count = 4096

        print(
            f"[Encrypted Client {self.client_id}] Initialized: {len(self.train_loader.dataset)} samples, "
            f"{self.total_params:,} parameters across {(self.total_params + self.slot_count - 1)//self.slot_count} CKKS chunks.",
            flush=True,
        )

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        flat_w, _ = extract_and_flatten_state_dict(self.model.state_dict())
        enc_chunks = encrypt_vector_chunks(self.ckks_context, flat_w, chunk_size=self.slot_count)
        return fl.common.GetParametersRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=fl.common.Parameters(tensors=enc_chunks, tensor_type="tenseal_ckks"),
        )

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        server_round = ins.config.get("server_round", "?")
        epochs = int(ins.config.get("local_epochs", LOCAL_EPOCHS))

        # 1. Ingest & Decrypt Aggregated Global Ciphertexts from Server
        if ins.parameters.tensors:
            t_dec_start = time.perf_counter()
            decrypted_flat = decrypt_vector_chunks(
                self.ckks_context, ins.parameters.tensors, self.total_params
            )
            restored_state = reconstruct_state_dict(decrypted_flat, self.metadata, device=self.device)
            self.model.load_state_dict(restored_state)
            dec_time = time.perf_counter() - t_dec_start
            print(
                f"[Encrypted Client {self.client_id}] Ingested & Decrypted global model ciphertext "
                f"({len(ins.parameters.tensors)} chunks) in {dec_time:.3f}s",
                flush=True,
            )

        # 2. Local Training on Institutional Patient Data
        start_time = time.perf_counter()
        print(
            f"\n[Encrypted Client {self.client_id}] Starting Private Local Training (Round {server_round}, {epochs} epochs)...",
            flush=True,
        )
        weight_l1_before = parameter_l1_norm(self.model)
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
        weight_l1_after = parameter_l1_norm(self.model)
        print(
            f"[Encrypted Client {self.client_id}] Training Complete ({duration_sec:.2f}s) - Loss: {loss:.5f}, Accuracy: {accuracy * 100:.2f}%",
            flush=True,
        )

        # 3. Client-Side Homomorphic Tensor Encryption
        t_enc_start = time.perf_counter()
        trained_flat, _ = extract_and_flatten_state_dict(self.model.state_dict())
        encrypted_chunks = encrypt_vector_chunks(self.ckks_context, trained_flat, chunk_size=self.slot_count)
        enc_duration = time.perf_counter() - t_enc_start

        payload_mb = sum(len(c) for c in encrypted_chunks) / (1024 * 1024)
        print(
            f"[Encrypted Client {self.client_id}] Encrypted {len(encrypted_chunks)} chunks ({payload_mb:.2f} MB) "
            f"in {enc_duration:.3f}s. Model transmitted over gRPC under CKKS encryption.",
            flush=True,
        )

        return fl.common.FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=fl.common.Parameters(tensors=encrypted_chunks, tensor_type="tenseal_ckks"),
            num_examples=len(self.train_loader.dataset),
            metrics={
                "loss": float(loss),
                "accuracy": float(accuracy),
                "latency_sec": round(float(duration_sec), 3),
                "encryption_sec": round(float(enc_duration), 3),
                "weight_l1_before": weight_l1_before,
                "weight_l1_after": weight_l1_after,
                "weight_l1_change": weight_l1_after - weight_l1_before,
            },
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        server_round = ins.config.get("server_round", "?")
        if ins.parameters.tensors:
            decrypted_flat = decrypt_vector_chunks(
                self.ckks_context, ins.parameters.tensors, self.total_params
            )
            restored_state = reconstruct_state_dict(decrypted_flat, self.metadata, device=self.device)
            self.model.load_state_dict(restored_state)

        loss, accuracy = test(self.model, self.test_loader, device=self.device)
        print(
            f"[Encrypted Client {self.client_id}] Local Evaluation (Round {server_round}) - Loss: {loss:.5f}, Accuracy: {accuracy * 100:.2f}%",
            flush=True,
        )
        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.test_loader.dataset),
            metrics={"accuracy": float(accuracy), "loss": float(loss)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMed Flower Client")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="Client identifier (0, 1, or 2)",
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
        default=MRI_BATCH_SIZE,
        help=f"3D MRI batch size (default: {MRI_BATCH_SIZE})",
    )
    parser.add_argument(
        "--encrypted",
        action="store_true",
        help="Enable TenSEAL CKKS homomorphic encryption for client parameter updates",
    )
    parser.add_argument(
        "--context-path",
        type=str,
        default=None,
        help="Path to private CKKS client context file (.seal)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"==================================================", flush=True)
    mode_tag = "ENCRYPTED (TenSEAL CKKS)" if args.encrypted else "STANDARD (Plaintext)"
    print(f"  FedMed Client {args.client_id} Starting [{mode_tag}]", flush=True)
    print(f"  Target Server: {args.server_address}", flush=True)
    print(f"==================================================", flush=True)

    train_loader, test_loader = get_client_mri_dataloaders(
        client_id=args.client_id,
        batch_size=args.batch_size,
        seed=RANDOM_SEED,
    )

    if args.encrypted:
        from src.config import RESULTS_DIR
        if args.context_path:
            client_ctx = load_context_from_file(args.context_path)
        else:
            client_ctx_path, _ = ensure_consortium_keys(RESULTS_DIR)
            client_ctx = load_context_from_file(client_ctx_path)

        client = EncryptedFedMedClient(
            client_id=args.client_id,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            ckks_context=client_ctx,
        )
        try:
            fl.client.start_client(
                server_address=args.server_address,
                client=client,
            )
        except Exception as e:
            print(f"[Client {args.client_id}] Communication error: {e}", file=sys.stderr, flush=True)
            sys.exit(1)
    else:
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
