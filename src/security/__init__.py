"""
FedMed Security & Privacy-Preserving Package
--------------------------------------------
Implements TenSEAL CKKS Homomorphic Encryption for Cross-Silo Federated Learning.
Supports client-side encryption, server-side ciphertext aggregation, and authorized decryption.
"""

from src.security.tenseal_context import (
    create_ckks_context,
    get_public_context,
    serialize_context,
    deserialize_context,
)
from src.security.encryption import (
    encrypt_update,
    decrypt_update,
    flatten_torch_model,
    unflatten_torch_model,
)
from src.security.encrypted_aggregator import EncryptedAggregator

__all__ = [
    "create_ckks_context",
    "get_public_context",
    "serialize_context",
    "deserialize_context",
    "encrypt_update",
    "decrypt_update",
    "flatten_torch_model",
    "unflatten_torch_model",
    "EncryptedAggregator",
]
