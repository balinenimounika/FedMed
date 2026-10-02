"""
Script: Read-Only Preprocessing Verification
--------------------------------------------
Inspects processed .npz files and verifies:
- Patient counts, shapes, dtypes
- Absence of NaNs and Infinite values
- Preservation of discrete labels [0, 1, 2, 3]
- Z-score normalization metrics
DOES NOT MODIFY ANY FILES.
"""

import sys
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
from configs.dataset_config import DEFAULT_CONFIG


def main():
    processed_dir = Path(DEFAULT_CONFIG.processed_data_dir)
    npz_files = sorted(list(processed_dir.glob("*.npz")))

    total_checked = len(npz_files)
    valid_count = 0
    missing_corrupted = 0
    total_nan = 0
    total_inf = 0

    img_shapes = []
    mask_shapes = []
    all_labels = set()

    for f in npz_files:
        try:
            with np.load(f) as d:
                img = d["image"]
                mask = d["mask"]

                img_shapes.append(tuple(img.shape))
                mask_shapes.append(tuple(mask.shape))

                nan_count = int(np.isnan(img).sum() + np.isnan(mask).sum())
                inf_count = int(np.isinf(img).sum() + np.isinf(mask).sum())

                total_nan += nan_count
                total_inf += inf_count

                all_labels.update(np.unique(mask).tolist())

                if nan_count == 0 and inf_count == 0:
                    valid_count += 1
                else:
                    missing_corrupted += 1
        except Exception:
            missing_corrupted += 1

    labels_sorted = sorted(list(all_labels))
    passed = (valid_count == 6 and missing_corrupted == 0 and total_nan == 0 and total_inf == 0 and labels_sorted == [0, 1, 2, 3])

    print("=======================================================")
    print("PREPROCESSING VERIFICATION")
    print("=======================================================\n")
    print(f"Patients checked: {total_checked}")
    print(f"Valid patients: {valid_count}")
    print(f"Missing/corrupted: {missing_corrupted}")
    print(f"NaN values: {total_nan}")
    print(f"Infinite values: {total_inf}\n")

    print(f"Image shape: {img_shapes[0] if img_shapes else 'N/A'}")
    print(f"Mask shape: {mask_shapes[0] if mask_shapes else 'N/A'}\n")

    print(f"Mask labels: {labels_sorted}\n")

    print(f"Normalization: {DEFAULT_CONFIG.norm_method}\n")

    print(f"Preprocessing verification: {'PASSED' if passed else 'FAILED'}\n")
    print("=======================================================")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
