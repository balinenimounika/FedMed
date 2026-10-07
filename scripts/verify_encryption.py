"""
FedMed Homomorphic Encryption (TenSEAL CKKS) Verification Suite
==============================================================
Validates:
1. CKKS Cryptographic Parameter Security & Key Isolation (Zero-Server-Knowledge).
2. PyTorch Tensor Flattening, Packing, and Exact Reconstruction.
3. Multi-Client Homomorphic FedAvg vs Plaintext FedAvg Numerical Accuracy.
4. Client/Server Latency, Throughput, and Ciphertext Wire Overhead.
5. Flower gRPC Parameters Transport Wire Simulation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
import numpy as np
import tenseal as ts
import torch
import torch.nn as nn

from src.encryption import (
    CKKSConfig,
    create_ckks_context,
    decrypt_vector_chunks,
    encrypt_vector_chunks,
    export_public_context,
    extract_and_flatten_state_dict,
    homomorphic_fedavg,
    load_context_from_bytes,
    reconstruct_state_dict,
    verify_zero_server_knowledge,
)
from src.model import MedicalCNN


def print_header(title: str) -> None:
    print(f"\n{'=' * 75}\n  {title}\n{'=' * 75}", flush=True)


def test_key_isolation_and_security(config: CKKSConfig) -> Tuple[ts.Context, ts.Context]:
    """Test 1: Validate parameter bounds and verify that server cannot decrypt."""
    print_header("TEST 1: Cryptographic Context & Zero-Knowledge Server Isolation")
    config.validate()
    print(f"[OK] Parameter Validation Passed:")
    print(f"     - Polynomial Modulus Degree (N): {config.poly_modulus_degree}")
    print(f"     - Coeff Modulus Bit Sizes:       {config.coeff_mod_bit_sizes} (Sum: {sum(config.coeff_mod_bit_sizes)} bits <= 218 max for 128-bit)")
    print(f"     - Global Scale Factor:           2^{config.global_scale_bit}")
    print(f"     - Slots per Ciphertext:          {config.slot_count}")

    # Generate client private context
    t0 = time.perf_counter()
    client_ctx = create_ckks_context(config)
    keygen_time = time.perf_counter() - t0
    print(f"[OK] Client Private Context Generated in {keygen_time:.3f}s")
    assert client_ctx.is_private(), "Client context must contain secret key"

    # Export public context for server
    t0 = time.perf_counter()
    public_bytes = export_public_context(client_ctx)
    server_ctx = load_context_from_bytes(public_bytes)
    export_time = time.perf_counter() - t0
    print(f"[OK] Public Context Exported in {export_time:.3f}s (Wire Size: {len(public_bytes)/1024:.1f} KB)")
    assert server_ctx.is_public(), "Server context must be public only"
    assert not server_ctx.has_secret_key(), "Server context must NOT contain secret key"

    # Test sample vector encryption and verify server cannot decrypt
    sample_vec = [0.12345, -0.6789, 42.0, -100.5]
    enc_sample = ts.ckks_vector(client_ctx, sample_vec).serialize()

    is_isolated = verify_zero_server_knowledge(server_ctx, enc_sample)
    assert is_isolated, "CRITICAL FAULT: Server context was able to decrypt ciphertext!"
    print(f"[PASS] Zero-Knowledge Isolation Verified: Server decryption strictly denied by TenSEAL engine.")

    return client_ctx, server_ctx


def test_tensor_packing_lossless_reconstruction(model: nn.Module) -> None:
    """Test 2: Verify PyTorch state_dict flattening and unflattening is lossless."""
    print_header("TEST 2: PyTorch Tensor Flattening & Lossless Unpacking")
    original_sd = model.state_dict()
    flat_weights, metadata = extract_and_flatten_state_dict(original_sd)

    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Total Tensors:      {len(metadata)}")
    print(f"Total Parameters:   {len(flat_weights):,}")

    reconstructed_sd = reconstruct_state_dict(flat_weights, metadata)
    assert len(original_sd) == len(reconstructed_sd), "Tensor count mismatch"

    max_diff = 0.0
    for key in original_sd:
        diff = torch.max(torch.abs(original_sd[key] - reconstructed_sd[key])).item()
        if diff > max_diff:
            max_diff = diff

    print(f"Max Reconstruction Difference: {max_diff}")
    assert max_diff == 0.0, f"Reconstruction was not strictly lossless! Max diff: {max_diff}"
    print("[PASS] Tensor Flattening and Unpacking is strictly bitwise lossless.")


def test_homomorphic_fedavg_numerical_accuracy(
    client_ctx: ts.Context,
    server_ctx: ts.Context,
    slot_count: int = 4096,
    num_clients: int = 3,
) -> None:
    """Test 3: Multi-client Homomorphic FedAvg vs Plaintext FedAvg."""
    print_header("TEST 3: Multi-Client HE-FedAvg Numerical Accuracy & Error Margins")

    # Instantiate base model and generate simulated client updates
    base_model = MedicalCNN()
    client_models: List[MedicalCNN] = []
    client_flats: List[np.ndarray] = []
    metadata = None

    # Institutional sample counts (non-IID distribution)
    sample_counts = [200, 300, 500]
    total_samples = sum(sample_counts)
    weights = [n / total_samples for n in sample_counts]

    print(f"Simulating {num_clients} Medical Institutions:")
    for i in range(num_clients):
        m = MedicalCNN()
        # Add random institutional weight perturbation
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.05 * (i + 1))
        flat, meta = extract_and_flatten_state_dict(m.state_dict())
        client_models.append(m)
        client_flats.append(flat)
        metadata = meta
        print(f"  - Client {i} (Hospital {chr(65+i)}): {sample_counts[i]} samples (Weight: {weights[i]:.2%})")

    total_params = len(client_flats[0])

    # 1. Plaintext FedAvg Ground Truth
    t0 = time.perf_counter()
    expected_flat = np.zeros(total_params, dtype=np.float64)
    for i in range(num_clients):
        expected_flat += weights[i] * client_flats[i]
    t_plain = time.perf_counter() - t0

    # 2. Client-Side Encryption
    client_ciphertexts: List[List[bytes]] = []
    enc_times: List[float] = []
    total_wire_bytes = 0

    print(f"\nClient-Side PyTorch Tensor Encryption (Slot size: {slot_count}):")
    for i in range(num_clients):
        t_start = time.perf_counter()
        enc_chunks = encrypt_vector_chunks(client_ctx, client_flats[i])
        t_elapsed = time.perf_counter() - t_start
        enc_times.append(t_elapsed)
        client_ciphertexts.append(enc_chunks)
        client_bytes = sum(len(c) for c in enc_chunks)
        total_wire_bytes += client_bytes
        print(f"  - Client {i}: Encrypted {len(enc_chunks)} chunks in {t_elapsed:.3f}s ({len(enc_chunks)/t_elapsed:.1f} chunks/s, {client_bytes/(1024*1024):.2f} MB)")

    # 3. Server-Side Homomorphic Aggregation (Ciphertext Space)
    print(f"\nServer-Side Ciphertext Aggregation (Public Context Only):")
    t0 = time.perf_counter()
    aggregated_cipher_chunks = homomorphic_fedavg(server_ctx, client_ciphertexts, weights)
    t_agg = time.perf_counter() - t0
    print(f"  - Aggregated {len(aggregated_cipher_chunks)} ciphertext chunks in {t_agg:.4f}s")

    # 4. Client-Side Decryption & Model Restoration
    print(f"\nClient-Side Decryption & Model State Restoration:")
    t0 = time.perf_counter()
    decrypted_flat = decrypt_vector_chunks(client_ctx, aggregated_cipher_chunks, total_params)
    t_dec = time.perf_counter() - t0
    print(f"  - Decrypted in {t_dec:.3f}s ({total_params/t_dec:,.0f} parameters/sec)")

    # 5. Accuracy & Error Metrics
    abs_errors = np.abs(decrypted_flat - expected_flat)
    mae = float(np.max(abs_errors))
    mean_abs_error = float(np.mean(abs_errors))
    mse = float(np.mean((decrypted_flat - expected_flat) ** 2))
    snr_db = 10 * np.log10(np.var(expected_flat) / (mse + 1e-15))

    print(f"\nNumerical Error Analysis (Decrypted vs Ground Truth Plaintext):")
    print(f"  - Maximum Absolute Error (MAE): {mae:.4e}")
    print(f"  - Mean Absolute Error:          {mean_abs_error:.4e}")
    print(f"  - Mean Squared Error (MSE):     {mse:.4e}")
    print(f"  - Signal-to-Noise Ratio (SNR):  {snr_db:.2f} dB")

    # Cryptographic precision bound: with Delta = 2^40, MAE must be < 1e-5
    assert mae < 1e-5, f"Accuracy error exceeds tolerance: MAE={mae:.2e}"
    assert mse < 1e-10, f"MSE error exceeds tolerance: MSE={mse:.2e}"
    print(f"[PASS] CKKS Approximated FedAvg conforms strictly to precision tolerances.")


def test_flower_grpc_wire_transport(
    client_ctx: ts.Context,
    server_ctx: ts.Context,
) -> None:
    """Test 4: Wire transport encapsulation inside Flower flwr.common.Parameters."""
    print_header("TEST 4: Flower gRPC Parameters Wire Representation")

    # Encrypt a test tensor vector
    test_weights = np.linspace(-1.0, 1.0, 4096, dtype=np.float64)
    enc_chunks = encrypt_vector_chunks(client_ctx, test_weights)

    # Wrap into Flower Parameters message
    t0 = time.perf_counter()
    flwr_params = fl.common.Parameters(
        tensors=enc_chunks,
        tensor_type="tenseal_ckks_v1",
    )
    pack_time = time.perf_counter() - t0

    print(f"[OK] Flower Parameters Object Created in {pack_time * 1e6:.1f} µs:")
    print(f"     - Tensor Type Tag: {flwr_params.tensor_type}")
    print(f"     - Chunks Count:    {len(flwr_params.tensors)}")
    print(f"     - Total Payload:   {sum(len(t) for t in flwr_params.tensors) / 1024:.1f} KB")

    # Simulate wire extraction by server
    extracted_chunks = flwr_params.tensors
    assert len(extracted_chunks) == len(enc_chunks)
    assert extracted_chunks[0] == enc_chunks[0]

    # Server performs scalar scaling over extracted chunks
    vec = ts.ckks_vector_from(server_ctx, extracted_chunks[0])
    scaled_vec = vec * 0.5
    res_bytes = scaled_vec.serialize()

    # Client receives and decrypts
    dec_vec = ts.ckks_vector_from(client_ctx, res_bytes)
    dec_plain = np.array(dec_vec.decrypt()[: len(test_weights)])

    mae = np.max(np.abs(dec_plain - (test_weights * 0.5)))
    assert mae < 1e-6, f"Wire transport round-trip failed: MAE={mae}"
    print(f"[PASS] Flower gRPC Wire Transport simulation passed with MAE: {mae:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMed TenSEAL Verification")
    parser.add_argument("--poly-modulus", type=int, default=8192, choices=[8192, 16384])
    args = parser.parse_args()

    config = CKKSConfig(
        poly_modulus_degree=args.poly_modulus,
        coeff_mod_bit_sizes=[60, 40, 40, 60] if args.poly_modulus == 8192 else [60, 40, 40, 40, 60],
        global_scale_bit=40,
        generate_galois_keys=False,
        generate_relin_keys=False,
    )

    print("=" * 75)
    print("      FedMed Privacy-Preserving AI: TenSEAL CKKS Verification Suite")
    print("=" * 75)

    client_ctx, server_ctx = test_key_isolation_and_security(config)
    test_tensor_packing_lossless_reconstruction(MedicalCNN())
    test_homomorphic_fedavg_numerical_accuracy(client_ctx, server_ctx, slot_count=config.slot_count, num_clients=3)
    test_flower_grpc_wire_transport(client_ctx, server_ctx)

    print("\n" + "=" * 75)
    print(" [ALL VERIFICATION TESTS PASSED SUCCESSFULLY]")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
