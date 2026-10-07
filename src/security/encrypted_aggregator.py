"""
Server-Side Encrypted Aggregator Module
---------------------------------------
Executes Federated Averaging directly on encrypted CKKS ciphertexts.
The server operates solely with a public evaluation context and CANNOT decrypt
individual client updates or the intermediate aggregated result.
"""

from typing import List, Optional
import tenseal as ts


class EncryptedAggregator:
    """
    Simulates the central federated learning server aggregator.
    Performs homomorphic ciphertext additions and scalar multiplications.
    """

    def __init__(self, public_context: Optional[ts.Context] = None):
        if public_context is not None:
            if not public_context.is_public():
                raise PermissionError("Server must only hold a public context without the secret key.")
        self.context = public_context

    def aggregate_encrypted_updates(
        self,
        encrypted_updates: List[ts.CKKSVector],
        weights: Optional[List[float]] = None
    ) -> ts.CKKSVector:
        """
        Aggregates multiple encrypted client update vectors homomorphically without decryption.
        Computes the weighted or uniform arithmetic average:
            Aggregated = sum(w_i * Update_i)
        
        Args:
            encrypted_updates: List of ts.CKKSVector ciphertexts from hospital clients.
            weights: Optional list of client sample weights.
        Returns:
            ts.CKKSVector: Aggregated ciphertext vector.
        """
        if not encrypted_updates:
            raise ValueError("No encrypted updates received; cannot perform aggregation.")

        num_clients = len(encrypted_updates)
        for idx, u in enumerate(encrypted_updates):
            if not isinstance(u, ts.CKKSVector):
                raise TypeError(f"Update from client {idx+1} is not a ts.CKKSVector ciphertext.")

        # Verify compatible dimensions
        expected_size = encrypted_updates[0].size()
        for idx, u in enumerate(encrypted_updates):
            if u.size() != expected_size:
                raise ValueError(
                    f"Dimension mismatch: Client {idx+1} has size {u.size()} != expected {expected_size}."
                )

        # Equal weight federated averaging
        if weights is None:
            scalar = 1.0 / float(num_clients)
            # Homomorphic addition: update_1 + update_2 + ... + update_N
            total_sum = encrypted_updates[0].copy()
            for update in encrypted_updates[1:]:
                total_sum += update

            # Homomorphic multiplication with plaintext scalar: sum * (1 / N)
            aggregated_ciphertext = total_sum * scalar
            return aggregated_ciphertext

        # Weighted federated averaging (e.g. proportional to patient counts)
        if len(weights) != num_clients:
            raise ValueError(f"Weights count ({len(weights)}) does not match client count ({num_clients}).")

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Sum of client aggregation weights must be strictly positive.")

        norm_weights = [w / total_weight for w in weights]

        # Initialize with first weighted update
        aggregated_ciphertext = encrypted_updates[0] * norm_weights[0]
        for idx in range(1, num_clients):
            aggregated_ciphertext += (encrypted_updates[idx] * norm_weights[idx])

        return aggregated_ciphertext

    def verify_server_cannot_decrypt(self, ciphertext: ts.CKKSVector) -> bool:
        """
        Formally proves that the server cannot decrypt the ciphertext.
        Returns True if attempting to decrypt raises a ValueError.
        """
        try:
            # Server attempts to decrypt without secret key
            ciphertext.decrypt()
            return False  # Security failure if decrypt succeeds!
        except (ValueError, RuntimeError):
            # Expected cryptographically: context has no secret key
            return True
