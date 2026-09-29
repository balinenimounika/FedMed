"""
MRI Dataset & DataLoader Module
-------------------------------
Memory-efficient PyTorch Dataset for 3D Brain MRI volumes.
Lazy-loads individual patient files from disk during iteration to prevent RAM exhaustion.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.dataset_config import DatasetConfig, DEFAULT_CONFIG


class MRIDataset(Dataset):
    """
    PyTorch Dataset for 3D MRI volumes and segmentation masks.
    Loads patient data lazily from .npz archive on disk.
    
    Returns per sample:
        image: torch.FloatTensor of shape (C, D, H, W) e.g. (4, 128, 128, 128)
        mask:  torch.LongTensor of shape (D, H, W) e.g. (128, 128, 128)
        patient_id: str
    """

    def __init__(
        self,
        patient_ids: Optional[List[str]] = None,
        split: Optional[str] = None,
        config: DatasetConfig = DEFAULT_CONFIG
    ):
        self.config = config
        self.processed_dir = Path(config.processed_data_dir)
        self.splits_dir = Path(config.splits_dir)

        # Determine patient IDs
        if patient_ids is not None:
            self.patient_ids = sorted(patient_ids)
        elif split is not None:
            split_file = self.splits_dir / f"{split}.json"
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}. Run preprocessing first.")
            with open(split_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                self.patient_ids = manifest["patient_ids"]
        else:
            # Load all available processed files in directory
            npz_files = list(self.processed_dir.glob("*.npz"))
            self.patient_ids = sorted([f.stem for f in npz_files])

        # Verify files exist on disk
        self.samples: List[Tuple[str, Path]] = []
        for pid in self.patient_ids:
            npz_path = self.processed_dir / f"{pid}.npz"
            if npz_path.exists():
                self.samples.append((pid, npz_path))
            else:
                print(f"[Warning] Missing processed file for patient: {pid} at {npz_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Lazily loads single patient .npz from disk into RAM for active batch.
        """
        pid, npz_path = self.samples[idx]

        with np.load(npz_path) as data:
            image_arr = data["image"]  # shape: (C, D, H, W), float32
            mask_arr = data["mask"]    # shape: (D, H, W), int64

        image_tensor = torch.from_numpy(image_arr).float()
        mask_tensor = torch.from_numpy(mask_arr).long()

        return image_tensor, mask_tensor, pid


def get_dataloader(
    split: str,
    batch_size: int = 1,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    config: DatasetConfig = DEFAULT_CONFIG
) -> DataLoader:
    """
    Convenience factory to create a DataLoader for a given split ('train', 'val', or 'test').
    """
    dataset = MRIDataset(split=split, config=config)
    is_train = (split.lower() == "train")
    do_shuffle = is_train if shuffle is None else shuffle

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader
