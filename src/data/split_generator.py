"""
Split Generator Module
----------------------
Partitions patient records into reproducible, leak-free Training, Validation, and Test sets.
Guarantees strict patient-level isolation (no slices or volumes from the same patient across splits).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from configs.dataset_config import DatasetConfig, DEFAULT_CONFIG


class SplitGenerator:
    """
    Partitions patient IDs into train, val, and test splits using a fixed random seed.
    Saves split manifests and guarantees zero patient leakage.
    """

    def __init__(self, config: DatasetConfig = DEFAULT_CONFIG):
        self.config = config
        self.splits_dir = Path(config.splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def create_splits(
        self,
        patient_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Splits list of patient IDs into train, val, test splits.
        """
        sorted_pids = sorted(list(set(patient_ids)))
        num_patients = len(sorted_pids)

        if num_patients == 0:
            raise ValueError("Cannot create splits from an empty list of patients.")

        # Fixed seed random generator
        rng = np.random.RandomState(self.config.random_seed)
        shuffled_pids = sorted_pids.copy()
        rng.shuffle(shuffled_pids)

        # Calculate split indices
        train_count = int(round(num_patients * self.config.train_ratio))
        val_count = int(round(num_patients * self.config.val_ratio))
        
        # Ensure at least 1 sample in each split if num_patients >= 3
        if num_patients >= 3:
            train_count = max(1, train_count)
            val_count = max(1, val_count)
            test_count = max(1, num_patients - train_count - val_count)
            # Re-adjust train if needed
            if train_count + val_count + test_count > num_patients:
                train_count = num_patients - val_count - test_count
        else:
            # Fallback for very small sample sets
            train_count = max(1, num_patients - 1)
            val_count = 1 if num_patients > 1 else 0
            test_count = 0

        train_pids = shuffled_pids[:train_count]
        val_pids = shuffled_pids[train_count:train_count + val_count]
        test_pids = shuffled_pids[train_count + val_count:]

        splits = {
            "train": sorted(train_pids),
            "val": sorted(val_pids),
            "test": sorted(test_pids)
        }

        # Check for leakage
        self.verify_no_leakage(splits)

        # Save manifests
        self.save_splits(splits)

        return splits

    @staticmethod
    def verify_no_leakage(splits: Dict[str, List[str]]) -> bool:
        """
        Verifies that sets of patient IDs are mutually exclusive.
        Raises AssertionError if any overlap is detected.
        """
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        tv_overlap = train_set.intersection(val_set)
        tt_overlap = train_set.intersection(test_set)
        vt_overlap = val_set.intersection(test_set)

        if tv_overlap:
            raise AssertionError(f"Leakage detected between Train and Val: {tv_overlap}")
        if tt_overlap:
            raise AssertionError(f"Leakage detected between Train and Test: {tt_overlap}")
        if vt_overlap:
            raise AssertionError(f"Leakage detected between Val and Test: {vt_overlap}")

        return True

    def save_splits(self, splits: Dict[str, List[str]]) -> None:
        """Saves split manifests to JSON files."""
        for split_name, pids in splits.items():
            out_file = self.splits_dir / f"{split_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"split": split_name, "count": len(pids), "patient_ids": pids}, f, indent=2)

        summary = {
            "random_seed": self.config.random_seed,
            "train_ratio": self.config.train_ratio,
            "val_ratio": self.config.val_ratio,
            "test_ratio": self.config.test_ratio,
            "counts": {k: len(v) for k, v in splits.items()},
            "total_patients": sum(len(v) for v in splits.values())
        }
        with open(self.splits_dir / "splits_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def load_splits(self) -> Dict[str, List[str]]:
        """Loads existing train/val/test split manifests from disk."""
        splits = {}
        for split_name in ["train", "val", "test"]:
            split_file = self.splits_dir / f"{split_name}.json"
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}. Run preprocessing first.")
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                splits[split_name] = data["patient_ids"]
        return splits
