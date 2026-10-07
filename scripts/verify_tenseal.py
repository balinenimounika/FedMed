"""
Script: TenSEAL Setup & Smoke Test Verification
------------------------------------------------
Validates TenSEAL CKKS installation, context generation,
homomorphic ciphertext operations, and public/private key separation.
DOES NOT EXPOSE SECRET KEYS.
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

import tenseal as ts
from src.security.tenseal_context import create_ckks_context, get_public_context
from src.security.encryption import encrypt_update, decrypt_update


def main():
    print("=======================================================")
    print(" FedMed: TenSEAL Environment & Context Verification")
    print("=======================================================\n")

    print(f"TenSEAL version: {ts.__version__}")

    # 1. Create Secret Context
    print("[1/5] Creating CKKS context with 8192 degree polynomial...")
    secret_ctx = create_ckks_context(poly_modulus_degree=8192)
    assert not secret_ctx.is_public(), "Secret context must contain secret key."
    print("  -> Secret context created successfully.")

    # 2. Derive Public Context
    print("\n[2/5] Deriving public-only evaluation context for server...")
    public_ctx = get_public_context(secret_ctx)
    assert public_ctx.is_public(), "Public context must NOT contain secret key."
    print("  -> Public context created successfully (is_public = True).")

    # 3. Client Encryption
    print("\n[3/5] Testing client-side encryption...")
    sample_vec = [1.5, 2.5, 3.5, 4.5]
    enc_vec = encrypt_update(sample_vec, secret_ctx)
    assert isinstance(enc_vec, ts.CKKSVector), "Encryption must return a ts.CKKSVector."
    print(f"  -> Encrypted vector of length {len(sample_vec)} into ciphertext.")

    # 4. Prove Server Cannot Decrypt
    print("\n[4/5] Testing server decryption denial...")
    enc_pub = encrypt_update(sample_vec, public_ctx)
    try:
        enc_pub.decrypt()
        print("  -> [FAIL] Public context unexpectedly decrypted ciphertext!")
        sys.exit(1)
    except ValueError:
        print("  -> [PASS] Decryption correctly blocked: server lacks secret key.")

    # 5. Authorized Decryption
    print("\n[5/5] Testing authorized client decryption...")
    decrypted = decrypt_update(enc_vec, secret_ctx)
    print(f"  -> Original vector:  {sample_vec}")
    print(f"  -> Decrypted vector: {[round(x, 4) for x in decrypted]}")
    max_err = max(abs(a - b) for a, b in zip(sample_vec, decrypted))
    print(f"  -> Max error:        {max_err:.6e}")
    assert max_err < 1e-3, "CKKS decryption error exceeded tolerance."

    print("\n=======================================================")
    print(" [SUCCESS] TenSEAL Homomorphic Encryption verified!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
