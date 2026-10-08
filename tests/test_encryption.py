"""
Unit tests for FedMed TenSEAL CKKS encryption module.
"""

import numpy as np
import pytest
import tenseal as ts
import torch

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


def test_ckks_config_security_bounds():
    """Verify security bounds checking according to HomomorphicEncryption.org."""
    # Secure config
    cfg_secure = CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    cfg_secure.validate()

    # Insecure config exceeding 218 bits for N=8192
    cfg_insecure = CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 60, 60, 60])
    with pytest.raises(ValueError, match="Insecure CKKS parameters"):
        cfg_insecure.validate()


def test_zero_knowledge_isolation():
    """Verify server context cannot decrypt any client ciphertext."""
    client_ctx = create_ckks_context()
    public_bytes = export_public_context(client_ctx)
    server_ctx = load_context_from_bytes(public_bytes)

    assert client_ctx.is_private()
    assert server_ctx.is_public()
    assert not server_ctx.has_secret_key()

    vec = ts.ckks_vector(client_ctx, [1.0, 2.0, 3.0])
    cipher_bytes = vec.serialize()

    # Server attempt to decrypt must be refused
    assert verify_zero_server_knowledge(server_ctx, cipher_bytes)


def test_tensor_flatten_and_reconstruct():
    """Verify state_dict flattening and reconstruction is bitwise lossless."""
    model = MedicalCNN()
    original_sd = model.state_dict()

    flat_array, metadata = extract_and_flatten_state_dict(original_sd)
    restored_sd = reconstruct_state_dict(flat_array, metadata)

    assert len(original_sd) == len(restored_sd)
    for k in original_sd:
        assert torch.equal(original_sd[k], restored_sd[k])


def test_homomorphic_fedavg_precision():
    """Verify ciphertext-space federated averaging matches plaintext reference."""
    client_ctx = create_ckks_context()
    server_ctx = load_context_from_bytes(export_public_context(client_ctx))

    v1 = np.array([0.1, -0.5, 1.25, 3.4], dtype=np.float64)
    v2 = np.array([0.9, 0.5, -0.25, 1.6], dtype=np.float64)

    w1, w2 = 0.3, 0.7
    expected = w1 * v1 + w2 * v2

    c1 = encrypt_vector_chunks(client_ctx, v1, chunk_size=4096)
    c2 = encrypt_vector_chunks(client_ctx, v2, chunk_size=4096)

    agg_chunks = homomorphic_fedavg(server_ctx, [c1, c2], [w1, w2])
    decrypted = decrypt_vector_chunks(client_ctx, agg_chunks, len(v1))

    np.testing.assert_allclose(decrypted, expected, atol=1e-5, rtol=1e-5)
