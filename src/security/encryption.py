"""
Client-Side Encryption & Authorized Decryption
---------------------------------------------
Implements encryption of local model update vectors using TenSEAL CKKS,
transmission serialization helpers, and authorized client-side decryption.
"""

from typing import List, Union, Optional
import numpy as np
import torch
import tenseal as ts


def encrypt_update(
    update_vector: Union[List[float], np.ndarray, torch.Tensor],
    context: ts.Context
) -> ts.CKKSVector:
    """
    Encrypts a client's model or gradient update vector using the CKKS scheme.
    Args:
        update_vector: numeric array, list, or tensor of update values.
        context: TenSEAL context (with encryption keys).
    Returns:
        ts.CKKSVector: encrypted ciphertext vector.
    """
    # Convert input to 1D float list
    if isinstance(update_vector, torch.Tensor):
        vector = update_vector.detach().cpu().flatten().numpy().astype(np.float64).tolist()
    elif isinstance(update_vector, np.ndarray):
        vector = update_vector.flatten().astype(np.float64).tolist()
    elif isinstance(update_vector, (list, tuple)):
        vector = [float(x) for x in update_vector]
    else:
        raise TypeError(f"Unsupported update vector type: {type(update_vector)}")

    if len(vector) == 0:
        raise ValueError("Cannot encrypt an empty update vector.")

    # Validate against NaN and Inf values
    arr = np.array(vector)
    if np.isnan(arr).any():
        raise ValueError("Update vector contains NaN values; cannot encrypt.")
    if np.isinf(arr).any():
        raise ValueError("Update vector contains infinite values; cannot encrypt.")

    ciphertext = ts.ckks_vector(context, vector)
    return ciphertext


def decrypt_update(
    ciphertext: ts.CKKSVector,
    secret_context: Optional[ts.Context] = None
) -> List[float]:
    """
    Decrypts an encrypted CKKS ciphertext back to plaintext floating-point numbers.
    Can only be performed by an authorized party possessing the secret key.
    """
    if not isinstance(ciphertext, ts.CKKSVector):
        raise TypeError(f"Expected ts.CKKSVector, got {type(ciphertext)}")

    if secret_context is not None:
        if secret_context.is_public():
            raise PermissionError("Provided context is public and lacks the secret key.")
        c_bytes = ciphertext.serialize()
        authorized_vector = ts.ckks_vector_from(secret_context, c_bytes)
        return authorized_vector.decrypt()

    return ciphertext.decrypt()


def serialize_ciphertext(ciphertext: ts.CKKSVector) -> bytes:
    """Serializes ciphertext vector into bytes for network communication."""
    return ciphertext.serialize()


def deserialize_ciphertext(ciphertext_bytes: bytes, context: ts.Context) -> ts.CKKSVector:
    """Reconstructs ciphertext vector from serialized bytes linked to a context."""
    return ts.ckks_vector_from(context, ciphertext_bytes)


def flatten_torch_model(model: torch.nn.Module) -> np.ndarray:
    """Flattens all trainable parameters of a PyTorch model into a 1D numpy vector."""
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p.detach().cpu().flatten().numpy())
    if not params:
        return np.array([], dtype=np.float64)
    return np.concatenate(params).astype(np.float64)


def unflatten_torch_model(vector: np.ndarray, model: torch.nn.Module) -> torch.nn.Module:
    """Reconstructs 1D parameter array back into PyTorch model layers."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                numel = p.numel()
                slice_arr = vector[offset:offset + numel].reshape(p.shape)
                p.copy_(torch.from_numpy(slice_arr).to(p.device, p.dtype))
                offset += numel
    return model
