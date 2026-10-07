"""
Script: Test Encrypted Federated Aggregation with 3 Simulated Hospitals
----------------------------------------------------------------------
Demonstrates homomorphic encrypted aggregation across 3 hospitals using TenSEAL CKKS:
  Hospital 1 update: [1.0, 2.0, 3.0, 4.0]
  Hospital 2 update: [2.0, 4.0, 6.0, 8.0]
  Hospital 3 update: [3.0, 6.0, 9.0, 12.0]
Expected Baseline:   [2.0, 4.0, 6.0, 8.0]
Server executes aggregation directly on ciphertexts without decryption.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import numpy as np
import tenseal as ts
from src.security.tenseal_context import create_ckks_context, get_public_context
from src.security.encryption import encrypt_update, decrypt_update
from src.security.encrypted_aggregator import EncryptedAggregator


def main():
    print("=======================================================")
    print(" FedMed: 3-Hospital Encrypted Aggregation Simulation")
    print("=======================================================\n")

    # 1. Setup Cryptographic Contexts
    print("[1/4] Initializing TenSEAL CKKS cryptographic contexts...")
    secret_ctx = create_ckks_context(poly_modulus_degree=8192)
    server_public_ctx = get_public_context(secret_ctx)
    print("  -> Client-side secret context generated (holds private key).")
    print("  -> Server-side public context derived (zero private keys).")

    # 2. Define Hospital Updates
    h1_update = [1.0, 2.0, 3.0, 4.0]
    h2_update = [2.0, 4.0, 6.0, 8.0]
    h3_update = [3.0, 6.0, 9.0, 12.0]

    print("\n[2/4] Simulating Client-Side Local Model Updates...")
    print(f"  Hospital 1 Update: {h1_update}")
    print(f"  Hospital 2 Update: {h2_update}")
    print(f"  Hospital 3 Update: {h3_update}")

    # Compute Plaintext Verification Baseline
    expected_baseline = [
        (a + b + c) / 3.0
        for a, b, c in zip(h1_update, h2_update, h3_update)
    ]
    print(f"  -> Plaintext Baseline Average: {expected_baseline}")

    # Client-side encryption
    c1 = encrypt_update(h1_update, secret_ctx)
    c2 = encrypt_update(h2_update, secret_ctx)
    c3 = encrypt_update(h3_update, secret_ctx)
    print("  -> All 3 hospital updates encrypted into CKKS ciphertexts.")

    # 3. Server-Side Ciphertext Aggregation (Network Transmission Simulation)
    print("\n[3/4] Server-Side Homomorphic Ciphertext Aggregation...")
    # Clients transmit serialized ciphertext over network; server loads with public context
    c1_server = ts.ckks_vector_from(server_public_ctx, c1.serialize())
    c2_server = ts.ckks_vector_from(server_public_ctx, c2.serialize())
    c3_server = ts.ckks_vector_from(server_public_ctx, c3.serialize())
    print("  -> Server received serialized ciphertexts and bound them to evaluation context.")

    server = EncryptedAggregator(public_context=server_public_ctx)

    # Prove server cannot decrypt client updates
    assert server.verify_server_cannot_decrypt(c1_server), "Security breach: Server was able to decrypt client 1!"
    assert server.verify_server_cannot_decrypt(c2_server), "Security breach: Server was able to decrypt client 2!"
    assert server.verify_server_cannot_decrypt(c3_server), "Security breach: Server was able to decrypt client 3!"
    print("  -> Confirmed: Server CANNOT decrypt any individual client ciphertext.")

    # Server performs homomorphic addition and scalar multiplication
    encrypted_aggregate = server.aggregate_encrypted_updates([c1_server, c2_server, c3_server])
    print("  -> Server computed homomorphic average directly on ciphertexts.")
    assert server.verify_server_cannot_decrypt(encrypted_aggregate), "Server cannot decrypt aggregated ciphertext."

    # 4. Authorized Decryption & Verification
    print("\n[4/4] Authorized Decryption & Numerical Verification...")
    decrypted_aggregate = decrypt_update(encrypted_aggregate, secret_context=secret_ctx)
    decrypted_rounded = [round(x, 4) for x in decrypted_aggregate]

    abs_errors = [abs(dec - exp) for dec, exp in zip(decrypted_aggregate, expected_baseline)]
    max_error = max(abs_errors)

    print(f"\nExpected result:")
    print(f"{expected_baseline}")
    print(f"\nDecrypted encrypted-aggregation result:")
    print(f"{decrypted_rounded}")
    print(f"\nMaximum absolute error:")
    print(f"{max_error:.6e}")

    tolerance = 1e-3
    assert max_error <= tolerance, f"Maximum error {max_error} exceeded tolerance {tolerance}!"

    print("\nNumerical verification:")
    print("PASSED")
    print("\n=======================================================")
    print(" [SUCCESS] 3-Hospital Encrypted Aggregation PASSED!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
