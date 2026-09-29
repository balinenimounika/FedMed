"""
Script 03: Verify Dataset & Generate Week 1 Verification Report
--------------------------------------------------------------
Runs exhaustive validation checks on preprocessed data and split manifests:
- Checks raw file integrity and patient correspondence
- Verifies PyTorch DataLoader instantiation and batch tensor flow
- Verifies absence of patient leakage across splits
- Reports shapes before/after preprocessing, detected label classes, and intensity statistics
- Outputs the required Week 1 Verification Report.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from configs.dataset_config import DEFAULT_CONFIG
from src.data.dataset_reader import DatasetReader
from src.data.split_generator import SplitGenerator
from src.data.mri_dataset import MRIDataset, get_dataloader


def main():
    config = DEFAULT_CONFIG
    raw_dir = Path(config.raw_data_dir)
    processed_dir = Path(config.processed_data_dir)
    splits_dir = Path(config.splits_dir)

    print("\n=======================================================")
    print(" FedMed: Week 1 - Dataset Verification & Pipeline Check")
    print("=======================================================\n")

    # 1. Discover raw patient stats
    reader = DatasetReader(raw_dir)
    valid_patients, invalid_patients = reader.discover_patients()

    total_patients_found = len(valid_patients) + len(invalid_patients)
    valid_count = len(valid_patients)
    invalid_count = len(invalid_patients)

    # 2. Check split manifests
    split_summary_file = splits_dir / "splits_summary.json"
    if not split_summary_file.exists():
        print(f"[ERROR] Split summary not found at {split_summary_file}.")
        print("Please run 'python scripts/02_preprocess_dataset.py' first.")
        sys.exit(1)

    with open(split_summary_file, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    split_gen = SplitGenerator(config)
    splits = split_gen.load_splits()
    SplitGenerator.verify_no_leakage(splits)

    # 3. Check preprocessing metadata
    meta_json_file = processed_dir / "preprocessing_metadata.json"
    if not meta_json_file.exists():
        print(f"[ERROR] Preprocessing metadata not found at {meta_json_file}.")
        sys.exit(1)

    with open(meta_json_file, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    # Gather representative shapes, labels, intensity stats
    sample_pid = next(iter(meta_dict.keys()))
    sample_meta = meta_dict[sample_pid]
    orig_shape = sample_meta["orig_img_shape"]
    proc_shape = sample_meta["processed_img_shape"]

    # Gather all unique labels across entire dataset
    all_unique_labels = set()
    for pid, pdata in meta_dict.items():
        all_unique_labels.update(pdata["unique_labels"])
    all_labels_sorted = sorted(list(all_unique_labels))

    # Aggregate intensity statistics
    modality_means = {m: [] for m in config.modalities}
    modality_stds = {m: [] for m in config.modalities}
    for pid, pdata in meta_dict.items():
        int_stats = pdata.get("intensity_stats", {})
        for m in config.modalities:
            if m in int_stats:
                modality_means[m].append(int_stats[m]["mean"])
                modality_stds[m].append(int_stats[m]["std"])

    # 4. PyTorch DataLoader Verification
    print("[Verification Check] Testing PyTorch DataLoader integration...")
    train_loader = get_dataloader(split="train", batch_size=1, shuffle=False, config=config)
    val_loader = get_dataloader(split="val", batch_size=1, shuffle=False, config=config)
    test_loader = get_dataloader(split="test", batch_size=1, shuffle=False, config=config)

    # Fetch one batch from train loader to verify tensor properties
    sample_batch_img, sample_batch_mask, batch_pid = next(iter(train_loader))

    assert isinstance(sample_batch_img, torch.Tensor), "Image is not a PyTorch Tensor"
    assert isinstance(sample_batch_mask, torch.Tensor), "Mask is not a PyTorch Tensor"
    assert sample_batch_img.dtype == torch.float32, f"Expected float32, got {sample_batch_img.dtype}"
    assert sample_batch_mask.dtype == torch.int64, f"Expected int64, got {sample_batch_mask.dtype}"
    assert len(sample_batch_img.shape) == 5, f"Expected (B, C, D, H, W) 5D tensor, got {sample_batch_img.shape}"
    assert len(sample_batch_mask.shape) == 4, f"Expected (B, D, H, W) 4D tensor, got {sample_batch_mask.shape}"

    print(f"  [PASS] PyTorch DataLoader successfully loaded batch from patient '{batch_pid[0]}'.")
    print(f"  [PASS] Batch Image Tensor Shape: {list(sample_batch_img.shape)} (B, Channels, Depth, Height, Width)")
    print(f"  [PASS] Batch Mask Tensor Shape:  {list(sample_batch_mask.shape)} (B, Depth, Height, Width)")
    print(f"  [PASS] Tensor Value Range: min={sample_batch_img.min().item():.3f}, max={sample_batch_img.max().item():.3f}")

    # Determine dataset origin description
    if any("BraTS20_Training" in pid for pid in meta_dict.keys()):
        dataset_name = "BraTS (Brain Tumor Segmentation Challenge)"
        dataset_source = "MICCAI BraTS / Medical Decathlon (NIfTI 3D Volumes)"
    else:
        dataset_name = "Brain MRI 3D Segmentation Dataset"
        dataset_source = str(raw_dir)

    print("\n=======================================================")
    print("      WEEK 1 DATASET PREPROCESSING VERIFICATION REPORT")
    print("=======================================================")
    print(f"Dataset:                  {dataset_name}")
    print(f"Dataset source:           {dataset_source}")
    print(f"Total patients:           {total_patients_found}")
    print(f"Valid patients:           {valid_count}")
    print(f"Training patients:        {len(splits['train'])} ({splits['train']})")
    print(f"Validation patients:      {len(splits['val'])} ({splits['val']})")
    print(f"Test patients:            {len(splits['test'])} ({splits['test']})")
    print(f"Missing/corrupted samples: {invalid_count}")
    print(f"Original image shape:     {tuple(orig_shape)}")
    print(f"Processed image shape:    {tuple(proc_shape)}")
    print(f"Segmentation labels:      {all_labels_sorted} -> {config.class_names}")
    print(f"Normalization method:     {config.norm_method} (Z-score on non-zero brain voxels)")
    print(f"PyTorch DataLoader Test:  PASSED (Batch Shape: {list(sample_batch_img.shape)})")
    print(f"Preprocessing status:     COMPLETED")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
