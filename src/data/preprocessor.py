"""
Volume Preprocessor Module
--------------------------
Handles 3D Brain MRI preprocessing:
1. Brain bounding box extraction (background removal).
2. Intensity normalization (non-zero z-score with outlier percentile clipping).
3. Consistent 3D spatial resampling (trilinear for images, nearest-neighbor for masks).
4. Segmentation mask class label remapping (BraTS [0, 1, 2, 4] -> [0, 1, 2, 3]).
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.ndimage import zoom

from configs.dataset_config import DatasetConfig, DEFAULT_CONFIG


class VolumePreprocessor:
    """
    Standardized preprocessor for 3D multimodal MRI volumes and masks.
    Ensures consistent dimensions, stable intensities, and preserved discrete label masks.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def process_patient(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes full preprocessing pipeline on a single patient scan.
        Args:
            image: np.ndarray of shape (C, D, H, W), float32
            mask: np.ndarray of shape (D, H, W), integer
        Returns:
            processed_image: np.ndarray of shape (C, target_D, target_H, target_W), float32
            processed_mask: np.ndarray of shape (target_D, target_H, target_W), int64
            metadata: dict with preprocessing statistics (shapes, crop box, intensity stats)
        """
        orig_img_shape = image.shape
        orig_mask_shape = mask.shape

        # Step 1: Crop to non-zero brain bounding box
        if self.config.crop_to_nonzero_brain:
            image, mask, bbox = self.crop_to_nonzero_brain(image, mask, margin=self.config.margin_voxels)
        else:
            bbox = None

        # Step 2: Remap mask labels if specified (e.g. BraTS 4 -> 3)
        if self.config.remap_labels:
            mask = self.remap_labels(mask)

        # Step 3: Resize spatial dimensions to target shape (e.g. 128, 128, 128)
        if self.config.target_shape is not None:
            image, mask = self.resize_to_target(image, mask, self.config.target_shape)

        # Step 4: Intensity normalization (Z-score on non-zero brain voxels per channel)
        image, intensity_stats = self.normalize_intensities(image)

        # Detect present mask labels
        unique_labels = sorted(np.unique(mask).tolist())

        metadata = {
            "orig_img_shape": orig_img_shape,
            "orig_mask_shape": orig_mask_shape,
            "processed_img_shape": image.shape,
            "processed_mask_shape": mask.shape,
            "bbox": bbox,
            "unique_labels": unique_labels,
            "intensity_stats": intensity_stats
        }

        return image.astype(np.float32), mask.astype(np.int64), metadata

    @staticmethod
    def crop_to_nonzero_brain(
        image: np.ndarray,
        mask: np.ndarray,
        margin: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int, int, int]]:
        """
        Finds bounding box around non-zero voxels across all modalities and crops.
        Eliminates empty dark air space outside patient's skull.
        """
        # Sum intensity across channels to find any non-zero voxel
        brain_mask = np.any(image > 0, axis=0)

        # If image is unexpectedly all zeros, return uncropped
        if not np.any(brain_mask):
            d, h, w = mask.shape
            return image, mask, (0, d, 0, h, 0, w)

        d_indices, h_indices, w_indices = np.where(brain_mask)

        d_min = max(0, int(d_indices.min()) - margin)
        d_max = min(image.shape[1], int(d_indices.max()) + 1 + margin)

        h_min = max(0, int(h_indices.min()) - margin)
        h_max = min(image.shape[2], int(h_indices.max()) + 1 + margin)

        w_min = max(0, int(w_indices.min()) - margin)
        w_max = min(image.shape[3], int(w_indices.max()) + 1 + margin)

        cropped_img = image[:, d_min:d_max, h_min:h_max, w_min:w_max]
        cropped_mask = mask[d_min:d_max, h_min:h_max, w_min:w_max]
        bbox = (d_min, d_max, h_min, h_max, w_min, w_max)

        return cropped_img, cropped_mask, bbox

    @staticmethod
    def remap_labels(mask: np.ndarray) -> np.ndarray:
        """
        Standardizes BraTS segmentation labels:
        Raw BraTS: 0=background, 1=necrotic/non-enhancing core, 2=edema, 4=enhancing tumor
        Standardized: label 4 is remapped to 3 so labels form contiguous set {0, 1, 2, 3}.
        """
        remapped = mask.copy()
        remapped[mask == 4] = 3
        return remapped

    @staticmethod
    def resize_to_target(
        image: np.ndarray,
        mask: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes 3D volume to target_shape (Depth, Height, Width).
        - Image channels: rescaled using linear/spline interpolation (order=1)
        - Mask: rescaled using NEAREST-NEIGHBOR interpolation (order=0) to preserve integer labels.
        """
        current_shape = image.shape[1:]  # (D, H, W)
        if current_shape == target_shape:
            return image, mask

        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]

        # Resize each modality channel
        num_channels = image.shape[0]
        resized_channels = []
        for c in range(num_channels):
            # order=1: trilinear interpolation
            channel_resized = zoom(image[c], zoom_factors, order=1, prefilter=False)
            resized_channels.append(channel_resized)

        resized_img = np.stack(resized_channels, axis=0)

        # Resize mask with order=0 (nearest-neighbor) to avoid corrupting discrete classes
        resized_mask = zoom(mask, zoom_factors, order=0, prefilter=False)

        return resized_img, resized_mask

    def normalize_intensities(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Z-score normalization over non-zero brain voxels per channel.
        Applies percentile clipping (0.5%, 99.5%) to remove scanner outlier artifacts.
        Background voxels remain 0.
        """
        norm_img = np.zeros_like(image, dtype=np.float32)
        channel_names = self.config.modalities
        stats: Dict[str, Dict[str, float]] = {}

        p_low, p_high = self.config.clip_percentiles

        for c in range(image.shape[0]):
            c_name = channel_names[c] if c < len(channel_names) else f"modality_{c}"
            channel_data = image[c]
            brain_voxels = channel_data[channel_data > 0]

            if len(brain_voxels) > 0:
                # Outlier clipping
                vmin, vmax = np.percentile(brain_voxels, [p_low, p_high])
                clipped_data = np.clip(channel_data, vmin, vmax)

                clipped_brain = clipped_data[channel_data > 0]
                mean_val = float(np.mean(clipped_brain))
                std_val = float(np.std(clipped_brain))

                if std_val > 1e-6:
                    normalized = (clipped_data - mean_val) / std_val
                else:
                    normalized = clipped_data - mean_val

                # Preserve 0 for air background
                normalized[channel_data <= 0] = 0.0
                norm_img[c] = normalized

                stats[c_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": float(np.min(clipped_brain)),
                    "max": float(np.max(clipped_brain)),
                    "norm_min": float(np.min(normalized[channel_data > 0])),
                    "norm_max": float(np.max(normalized[channel_data > 0]))
                }
            else:
                norm_img[c] = channel_data
                stats[c_name] = {
                    "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "norm_min": 0.0, "norm_max": 0.0
                }

        return norm_img, stats
