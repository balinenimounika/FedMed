"""
TenSEAL CKKS Context Manager
----------------------------
Configures and manages Homomorphic Encryption contexts using the CKKS scheme.
Separates the secret context (held only by authorized clients/decryptors)
from the public evaluation context (shared with the untrusted aggregation server).
"""

from typing import List, Optional
import tenseal as ts


def create_ckks_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Optional[List[int]] = None,
    global_scale: float = 2**40
) -> ts.Context:
    """
    Creates a full TenSEAL CKKS context with secret key and Galois keys.
    This context is strictly confidential and must remain with authorized clients.
    """
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale
    context.generate_galois_keys()

    return context


def get_public_context(secret_context: ts.Context) -> ts.Context:
    """
    Derives a public-only evaluation context stripped of the secret key.
    Safe to transmit to the untrusted federated server.
    Attempting to decrypt using this context will raise a ValueError.
    """
    public_bytes = secret_context.serialize(save_secret_key=False)
    public_context = ts.context_from(public_bytes)
    return public_context


def serialize_context(context: ts.Context, save_secret_key: bool = False) -> bytes:
    """Serializes the context into bytes for network transmission or storage."""
    return context.serialize(save_secret_key=save_secret_key)


def deserialize_context(context_bytes: bytes) -> ts.Context:
    """Reconstructs a TenSEAL context from serialized bytes."""
    return ts.context_from(context_bytes)
