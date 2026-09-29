"""
FedMed Dataset Configuration
----------------------------
Central configuration for Brain MRI (BraTS / Medical Decathlon) dataset preprocessing,
including target dimensions, intensity normalization, directory paths, and train/val/test splits.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List

# Workspace Root
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class DatasetConfig:
    # Directory paths
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    splits_dir: Path = BASE_DIR / "data" / "splits"
    visualizations_dir: Path = BASE_DIR / "data" / "visualizations"
    
    # Supported modalities (BraTS standard)
    modalities: List[str] = field(default_factory=lambda: ["flair", "t1", "t1ce", "t2"])
    
    # Target volume dimensions (Depth, Height, Width) for 3D U-Net
    # (128, 128, 128) is standard for memory efficiency on single-GPU/CPU training
    target_shape: Tuple[int, int, int] = (128, 128, 128)
    
    # Background removal & cropping
    crop_to_nonzero_brain: bool = True
    margin_voxels: int = 4
    
    # Intensity Normalization
    # 'zscore_nonzero': compute mean & std only across non-zero brain voxels per channel
    # 'minmax_nonzero': min-max normalize non-zero voxels to [0, 1]
    norm_method: str = "zscore_nonzero"
    clip_percentiles: Tuple[float, float] = (0.5, 99.5)  # Suppress MRI outlier spikes
    
    # Dataset Split Ratios (Patient-level, leak-free)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Mask label remapping:
    # BraTS 2020 labels: 0=background, 1=necrotic/non-enhancing core, 2=edema, 4=enhancing tumor
    # Remap 4 -> 3 so labels are contiguous [0, 1, 2, 3] for standard PyTorch CrossEntropy / Dice
    remap_labels: bool = True
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: [
        "Background",
        "Necrotic/Non-Enhancing Core (NCR/NET)",
        "Peritumoral Edema (ED)",
        "Enhancing Tumor (ET)"
    ])

# Default singleton instance
DEFAULT_CONFIG = DatasetConfig()
