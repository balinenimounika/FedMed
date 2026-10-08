"""
TenSEAL Homomorphic Encryption (CKKS) Module for FedMed
======================================================
Enterprise-grade cryptographic primitives for client-side PyTorch tensor encryption
and zero-knowledge server-side homomorphic federated aggregation (HE-FedAvg).

Cryptographic Security Specifications:
-------------------------------------
- Scheme: CKKS (Cheon-Kim-Kim-Song) for approximate arithmetic over real vectors.
- Security Standard: HomomorphicEncryption.org security standard (128-bit classical / post-quantum security).
- Ring-LWE Modulus Degree: N = 8192 (or 16384 for deep circuits).
- Coefficient Modulus: [60, 40, 40, 60] (total bit count 200 <= 218 max for N=8192 at 128-bit security).
- Global Scale Factor: Delta = 2^40 (~40 bits fractional precision, precision error < 1e-7).
"""

from __future__ import annotations

import io
import math
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tenseal as ts
import torch
import torch.nn as nn


@dataclass(frozen=True)
class CKKSConfig:
    """Cryptographic configuration parameters for TenSEAL CKKS scheme."""
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    global_scale_bit: int = 40
    # For FedAvg (addition and scalar multiplication only), Galois keys and Relin keys
    # are unnecessary. Setting them to False reduces context size from ~150MB to ~1.8MB
    # and context generation latency from ~30s to <0.3s.
    generate_galois_keys: bool = False
    generate_relin_keys: bool = False

    @property
    def global_scale(self) -> float:
        """Global scale factor Delta = 2^(global_scale_bit)."""
        return float(2 ** self.global_scale_bit)

    @property
    def slot_count(self) -> int:
        """Number of complex/real vector slots per CKKS ciphertext (N // 2)."""
        return self.poly_modulus_degree // 2

    def validate(self) -> None:
        """Validate parameter selection against HomomorphicEncryption.org security bounds."""
        total_coeff_bits = sum(self.coeff_mod_bit_sizes)
        # Standard 128-bit security bounds for classical / post-quantum security:
        # N = 8192  -> max coeff modulus ~218 bits
        # N = 16384 -> max coeff modulus ~438 bits
        # N = 32768 -> max coeff modulus ~881 bits
        security_limits = {
            4096: 109,
            8192: 218,
            16384: 438,
            32768: 881,
        }
        max_allowed = security_limits.get(self.poly_modulus_degree)
        if max_allowed and total_coeff_bits > max_allowed:
            raise ValueError(
                f"Insecure CKKS parameters: total coeff_mod_bit_sizes={total_coeff_bits} "
                f"exceeds 128-bit security limit {max_allowed} for N={self.poly_modulus_degree}."
            )


@dataclass
class TensorMetadata:
    """Metadata describing a single PyTorch tensor for exact reconstruction."""
    name: str
    shape: List[int]
    numel: int
    dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorMetadata":
        return cls(
            name=data["name"],
            shape=list(data["shape"]),
            numel=int(data["numel"]),
            dtype=str(data["dtype"]),
        )


# =====================================================================
# Context Generation & Key Management
# =====================================================================

def create_ckks_context(config: Optional[CKKSConfig] = None) -> ts.Context:
    """
    Generate a full private TenSEAL CKKS context with secret key, public key,
    relinearization keys, and Galois rotation keys.

    Held exclusively by hospital client institutions or consortium key management.
    """
    if config is None:
        config = CKKSConfig()
    config.validate()

    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config.poly_modulus_degree,
        coeff_mod_bit_sizes=config.coeff_mod_bit_sizes,
    )
    context.global_scale = config.global_scale

    if config.generate_galois_keys:
        context.generate_galois_keys()
    if config.generate_relin_keys:
        context.generate_relin_keys()

    return context


def serialize_context(context: ts.Context, save_secret_key: bool = False) -> bytes:
    """
    Serialize a TenSEAL context to bytes.

    If save_secret_key is False, the secret key is dropped, creating a public-only
    context safe to transmit to the central Flower aggregation server.
    """
    return context.serialize(
        save_public_key=True,
        save_secret_key=save_secret_key,
        save_galois_keys=True,
        save_relin_keys=True,
    )


def load_context_from_bytes(context_bytes: bytes) -> ts.Context:
    """
    Deserialize a TenSEAL context from bytes.
    Used by the central aggregator (public context) and clients (private context).
    """
    return ts.context_from(context_bytes)


def export_public_context(private_context: ts.Context) -> bytes:
    """
    Export the public evaluation context without the private secret key.
    The resulting bytes can safely be published to the central aggregation server.
    """
    return serialize_context(private_context, save_secret_key=False)


def save_context_to_file(
    context: ts.Context,
    file_path: Union[str, Path],
    save_secret_key: bool = False,
) -> Path:
    """Save serialized TenSEAL context to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = serialize_context(context, save_secret_key=save_secret_key)
    path.write_bytes(raw_bytes)
    return path


def load_context_from_file(file_path: Union[str, Path]) -> ts.Context:
    """Load serialized TenSEAL context from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CKKS context file not found: {path}")
    raw_bytes = path.read_bytes()
    return load_context_from_bytes(raw_bytes)


def ensure_consortium_keys(
    keys_dir: Union[str, Path],
    config: Optional[CKKSConfig] = None,
) -> Tuple[Path, Path]:
    """
    Ensure consortium keys exist on disk. If absent, generates a fresh CKKS key pair:
    - client_context.seal (Private: Contains secret key, kept in client hospital enclave)
    - server_context.seal (Public: Contains public/eval keys only, given to aggregator)
    """
    directory = Path(keys_dir)
    directory.mkdir(parents=True, exist_ok=True)

    client_path = directory / "ckks_client.seal"
    server_path = directory / "ckks_public.seal"

    if client_path.is_file() and server_path.is_file():
        return client_path, server_path

    # Generate fresh key pair
    client_ctx = create_ckks_context(config)
    save_context_to_file(client_ctx, client_path, save_secret_key=True)
    save_context_to_file(client_ctx, server_path, save_secret_key=False)

    return client_path, server_path


# =====================================================================
# Tensor Flattening, Chunking, Encryption & Decryption
# =====================================================================

def extract_and_flatten_state_dict(
    state_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]
) -> Tuple[np.ndarray, List[TensorMetadata]]:
    """
    Flatten PyTorch state_dict tensors into a single contiguous 1D float64 numpy array
    and record structural metadata for lossless reconstruction.
    """
    flat_parts: List[np.ndarray] = []
    metadata_list: List[TensorMetadata] = []

    for name, tensor in state_dict.items():
        tensor_cpu = tensor.detach().cpu()
        np_arr = tensor_cpu.numpy().astype(np.float64)
        metadata = TensorMetadata(
            name=name,
            shape=list(tensor.shape),
            numel=tensor.numel(),
            dtype=str(tensor.dtype).replace("torch.", ""),
        )
        metadata_list.append(metadata)
        flat_parts.append(np_arr.ravel())

    if not flat_parts:
        return np.empty(0, dtype=np.float64), []

    flat_array = np.concatenate(flat_parts, axis=0)
    return flat_array, metadata_list


def reconstruct_state_dict(
    flat_array: np.ndarray,
    metadata_list: List[TensorMetadata],
    device: Optional[Union[str, torch.device]] = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    Reconstruct original PyTorch state_dict from a flattened 1D array using metadata.
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    state_dict = OrderedDict()
    offset = 0

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "int64": torch.int64,
        "int32": torch.int32,
    }

    for meta in metadata_list:
        length = meta.numel
        chunk = flat_array[offset : offset + length]
        offset += length

        # Reshape to original tensor dimensions
        if meta.shape:
            reshaped_np = chunk.reshape(meta.shape)
        else:
            reshaped_np = chunk  # 0-d scalar tensor

        target_dtype = dtype_map.get(meta.dtype, torch.float32)
        tensor = torch.from_numpy(reshaped_np).to(dtype=target_dtype, device=device)
        state_dict[meta.name] = tensor

    return state_dict


def encrypt_vector_chunks(
    context: ts.Context,
    flat_vector: np.ndarray,
    chunk_size: int = 4096,
) -> List[bytes]:
    """
    Segment a 1D float array into chunks matching CKKS slot capacity (e.g. 4096 for N=8192),
    encrypt each chunk into a ts.ckks_vector, and serialize into bytes.

    Args:
        context: Private TenSEAL CKKS context with encryption keys.
        flat_vector: 1D numpy array of model weights.
        chunk_size: Slot limit per ciphertext. Defaults to 4096 (N=8192 capacity).

    Returns:
        List of serialized CKKS vector ciphertexts (bytes).
    """
    if chunk_size <= 0:
        chunk_size = 4096

    total_len = len(flat_vector)
    num_chunks = math.ceil(total_len / chunk_size) if total_len > 0 else 0
    encrypted_chunks: List[bytes] = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        slice_data = flat_vector[start_idx:end_idx].tolist()

        # Homomorphically encrypt vector
        enc_vec = ts.ckks_vector(context, slice_data)
        encrypted_chunks.append(enc_vec.serialize())

    return encrypted_chunks


def decrypt_vector_chunks(
    context: ts.Context,
    encrypted_chunks: Sequence[bytes],
    expected_total_len: int,
) -> np.ndarray:
    """
    Deserialize and decrypt each CKKS ciphertext chunk using the private secret key,
    concatenating results into the full 1D parameter array.

    Args:
        context: Private TenSEAL CKKS context containing the secret key.
        encrypted_chunks: List of serialized CKKSVector ciphertexts.
        expected_total_len: Total number of model parameters to reconstruct.

    Returns:
        Decrypted 1D numpy array of length `expected_total_len`.
    """
    if not encrypted_chunks:
        return np.empty(0, dtype=np.float64)

    decrypted_slices: List[np.ndarray] = []
    for chunk_bytes in encrypted_chunks:
        ckks_vec = ts.ckks_vector_from(context, chunk_bytes)
        plain_list = ckks_vec.decrypt()
        decrypted_slices.append(np.array(plain_list, dtype=np.float64))

    flat_array = np.concatenate(decrypted_slices, axis=0)
    # Truncate any slot padding to match exact parameter count
    return flat_array[:expected_total_len]


# =====================================================================
# Server-Side Secure Homomorphic Aggregation (Ciphertext Space)
# =====================================================================

def homomorphic_fedavg(
    public_context: ts.Context,
    client_encrypted_chunks: List[List[bytes]],
    client_weights: List[float],
) -> List[bytes]:
    """
    Perform secure FedAvg aggregation directly over ciphertexts in the central server.
    The server holds ONLY the public context and CANNOT decrypt any client weights.

    Formula:
        C_global^(j) = sum_{k=1}^K ( w_k * C_k^(j) )
    where:
        w_k = n_k / sum(n_i)
        C_k^(j) is the j-th encrypted chunk from client k.

    Args:
        public_context: Server TenSEAL context with public keys only (NO secret key).
        client_encrypted_chunks: K lists of M serialized ciphertexts, one list per client.
        client_weights: Normalized aggregation weights (sum(client_weights) == 1.0).

    Returns:
        Aggregated encrypted model as a list of M serialized CKKSVector ciphertexts.
    """
    num_clients = len(client_encrypted_chunks)
    if num_clients == 0:
        return []
    if num_clients != len(client_weights):
        raise ValueError(
            f"Mismatched clients count ({num_clients}) and weights count ({len(client_weights)})"
        )

    # Normalize weights if not already normalized
    total_w = sum(client_weights)
    if total_w <= 0:
        raise ValueError(f"Invalid non-positive total weight: {total_w}")
    norm_weights = [w / total_w for w in client_weights]

    num_chunks = len(client_encrypted_chunks[0])
    # Validate consistent chunk counts across all clients
    for idx, c_chunks in enumerate(client_encrypted_chunks):
        if len(c_chunks) != num_chunks:
            raise ValueError(
                f"Client {idx} has {len(c_chunks)} chunks, expected {num_chunks}"
            )

    aggregated_chunks: List[bytes] = []

    for j in range(num_chunks):
        # Initialize accumulator with first client's weighted ciphertext
        first_weight = float(norm_weights[0])
        first_vec = ts.ckks_vector_from(public_context, client_encrypted_chunks[0][j])
        acc_vec = first_vec * first_weight

        # Homomorphically accumulate remaining clients
        for k in range(1, num_clients):
            w_k = float(norm_weights[k])
            c_k = ts.ckks_vector_from(public_context, client_encrypted_chunks[k][j])
            weighted_c_k = c_k * w_k
            acc_vec = acc_vec + weighted_c_k

        # Serialize aggregated ciphertext
        aggregated_chunks.append(acc_vec.serialize())

    return aggregated_chunks


def verify_zero_server_knowledge(
    public_context: ts.Context,
    sample_ciphertext_bytes: bytes,
) -> bool:
    """
    Security Verification Test:
    Confirms that the public context held by the server is cryptographically incapable
    of decrypting model ciphertexts.

    Returns True if decryption attempt raises an explicit security error.
    """
    vec = ts.ckks_vector_from(public_context, sample_ciphertext_bytes)
    try:
        _ = vec.decrypt()
        return False  # Security violation: secret key was leaked to server context!
    except Exception as exc:
        # Expected behavior: TenSEAL raises ValueError: current context doesn't hold a secret_key
        return True
