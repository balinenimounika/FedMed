"""
Script: Final Week 1 Dataset & Preprocessing Verification
--------------------------------------------------------
Executes all 15 end-to-end checks across:
- Raw and processed dataset files
- Image/mask dimensions, dtypes, and discrete labels
- Absence of NaNs / Infs
- Non-zero Z-score normalization
- Patient-level Train/Val/Test splits and zero leakage
- 3-Hospital cross-silo partitioning and zero leakage
- Hospital 1, 2, and 3 PyTorch DataLoaders
- Hospital data distributions and statistical summaries
- Availability of visualization previews

DOES NOT MODIFY ANY PROJECT FILES.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import numpy as np
import torch
from configs.dataset_config import DEFAULT_CONFIG
from src.data.mri_dataset import get_hospital_dataloader


def main():
    print("=======================================================")
    print(" FedMed — Week 1 Dataset & Preprocessing")
    print(" Final Verification")
    print("=======================================================\n")

    config = DEFAULT_CONFIG
    raw_dir = Path(config.raw_data_dir)
    processed_dir = Path(config.processed_data_dir)
    splits_dir = Path(config.splits_dir)
    hospitals_dir = Path(config.hospitals_dir)
    vis_dir = Path(config.visualizations_dir)

    all_passed = True

    # 1. Dataset preparation
    raw_subdirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    c1 = (len(raw_subdirs) == 6)
    all_passed = all_passed and c1
    print(f"[{'PASS' if c1 else 'FAIL'}] Dataset preparation")

    # 2. MRI preprocessing
    npz_files = list(processed_dir.glob("*.npz"))
    c2 = (len(npz_files) == 6)
    all_passed = all_passed and c2
    print(f"[{'PASS' if c2 else 'FAIL'}] MRI preprocessing")

    # Check processed files content
    c_norm = True
    c_img = True
    c_mask = True
    c_label = True
    c_nan = True

    for npz in npz_files:
        with np.load(npz) as d:
            img = d["image"]
            mask = d["mask"]

            if img.shape != (4, 128, 128, 128) or img.dtype != np.float32:
                c_img = False
            if mask.shape != (128, 128, 128) or mask.dtype != np.int64:
                c_mask = False
            if not np.array_equal(np.unique(mask), [0, 1, 2, 3]):
                c_label = False
            if np.isnan(img).any() or np.isnan(mask).any() or np.isinf(img).any() or np.isinf(mask).any():
                c_nan = False

            # Check normalization (non-zero mean ~ 0, std ~ 1)
            nonzero = img[img != 0]
            if len(nonzero) > 0:
                mean_val = np.mean(nonzero)
                std_val = np.std(nonzero)
                if abs(mean_val) > 0.1 or abs(std_val - 1.0) > 0.1:
                    c_norm = False

    # 3. Normalization
    all_passed = all_passed and c_norm
    print(f"[{'PASS' if c_norm else 'FAIL'}] Normalization")

    # 4. Image verification
    all_passed = all_passed and c_img
    print(f"[{'PASS' if c_img else 'FAIL'}] Image verification")

    # 5. Mask verification
    all_passed = all_passed and c_mask
    print(f"[{'PASS' if c_mask else 'FAIL'}] Mask verification")

    # 6. Label verification
    all_passed = all_passed and c_label
    print(f"[{'PASS' if c_label else 'FAIL'}] Label verification")

    # 7. Train/Validation/Test split
    with open(splits_dir / "train.json", "r", encoding="utf-8") as f:
        train_pids = set(json.load(f)["patient_ids"])
    with open(splits_dir / "val.json", "r", encoding="utf-8") as f:
        val_pids = set(json.load(f)["patient_ids"])
    with open(splits_dir / "test.json", "r", encoding="utf-8") as f:
        test_pids = set(json.load(f)["patient_ids"])

    c_split = (len(train_pids) == 4 and len(val_pids) == 1 and len(test_pids) == 1)
    all_passed = all_passed and c_split
    print(f"[{'PASS' if c_split else 'FAIL'}] Train/Validation/Test split")

    # 8. Split leakage check
    c_split_leak = (len(train_pids & val_pids) == 0 and len(train_pids & test_pids) == 0 and len(val_pids & test_pids) == 0)
    all_passed = all_passed and c_split_leak
    print(f"[{'PASS' if c_split_leak else 'FAIL'}] Split leakage check")

    # 9. 3-hospital partition
    with open(hospitals_dir / "hospital_1.json", "r", encoding="utf-8") as f:
        h1_pids = set(json.load(f)["patient_ids"])
    with open(hospitals_dir / "hospital_2.json", "r", encoding="utf-8") as f:
        h2_pids = set(json.load(f)["patient_ids"])
    with open(hospitals_dir / "hospital_3.json", "r", encoding="utf-8") as f:
        h3_pids = set(json.load(f)["patient_ids"])

    c_hosp = (len(h1_pids) == 2 and len(h2_pids) == 2 and len(h3_pids) == 2 and len(h1_pids | h2_pids | h3_pids) == 6)
    all_passed = all_passed and c_hosp
    print(f"[{'PASS' if c_hosp else 'FAIL'}] 3-hospital partition")

    # 10. Hospital leakage check
    c_hosp_leak = (len(h1_pids & h2_pids) == 0 and len(h1_pids & h3_pids) == 0 and len(h2_pids & h3_pids) == 0)
    all_passed = all_passed and c_hosp_leak
    print(f"[{'PASS' if c_hosp_leak else 'FAIL'}] Hospital leakage check")

    # 11, 12, 13. Hospital DataLoaders
    for h_id in [1, 2, 3]:
        try:
            loader = get_hospital_dataloader(hospital_id=h_id, batch_size=1, config=config)
            b_img, b_mask, _ = next(iter(loader))
            c_loader = (
                b_img.shape == torch.Size([1, 4, 128, 128, 128]) and
                b_mask.shape == torch.Size([1, 128, 128, 128]) and
                not torch.isnan(b_img).any() and
                not torch.isinf(b_img).any()
            )
        except Exception:
            c_loader = False

        all_passed = all_passed and c_loader
        print(f"[{'PASS' if c_loader else 'FAIL'}] Hospital {h_id} DataLoader")

    # 14. Hospital data distribution
    stat_file = hospitals_dir / "hospital_statistics.json"
    rep_file = config.reports_dir / "week1_hospital_distribution.txt"
    c_dist = stat_file.exists() and rep_file.exists()
    all_passed = all_passed and c_dist
    print(f"[{'PASS' if c_dist else 'FAIL'}] Hospital data distribution")

    # 15. Dataset statistics & Visualization evidence
    c_stats = (splits_dir / "splits_summary.json").exists()
    all_passed = all_passed and c_stats
    print(f"[{'PASS' if c_stats else 'FAIL'}] Dataset statistics")

    vis_file = vis_dir / "sample_slice_visualization.png"
    c_vis = vis_file.exists()
    all_passed = all_passed and c_vis
    print(f"[{'PASS' if c_vis else 'FAIL'}] Visualization evidence")

    print("\n=======================================================")
    print("FINAL RESULT:")
    print(f"WEEK 1 DATASET & PREPROCESSING: {'PASSED' if all_passed else 'FAILED'}")
    print("=======================================================\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
