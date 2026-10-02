"""
Script: Read-Only Hospital DataLoader Verification
--------------------------------------------------
Instantiates PyTorch DataLoaders for Hospital 1, Hospital 2, and Hospital 3.
Verifies batch loading, tensor shapes, dtypes, and absence of NaNs/Infs.
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

import torch
from configs.dataset_config import DEFAULT_CONFIG
from src.data.mri_dataset import get_hospital_dataloader


def main():
    print("=======================================================")
    print("HOSPITAL DATALOADER VERIFICATION")
    print("=======================================================\n")

    all_passed = True

    for h_id in [1, 2, 3]:
        loader = get_hospital_dataloader(hospital_id=h_id, batch_size=1, shuffle=False, config=DEFAULT_CONFIG)
        patient_count = len(loader.dataset)

        # Fetch first batch
        try:
            batch_img, batch_mask, pid = next(iter(loader))

            # Shape and type checks
            img_shape = list(batch_img.shape)
            mask_shape = list(batch_mask.shape)
            has_nan = torch.isnan(batch_img).any().item() or torch.isnan(batch_mask).any().item()
            has_inf = torch.isinf(batch_img).any().item() or torch.isinf(batch_mask).any().item()

            shape_valid = (
                len(img_shape) == 5 and img_shape[1] == 4 and img_shape[2:] == [128, 128, 128] and
                len(mask_shape) == 4 and mask_shape[1:] == [128, 128, 128] and
                batch_img.dtype == torch.float32 and
                batch_mask.dtype == torch.int64 and
                not has_nan and not has_inf
            )
            status_str = "PASSED" if shape_valid else "FAILED"
            if not shape_valid:
                all_passed = False

            b = img_shape[0]
            print(f"Hospital {h_id}:")
            print(f"    Patients: {patient_count}")
            print(f"    DataLoader: {status_str}")
            print(f"    Image batch shape: [{b},4,128,128,128]")
            print(f"    Mask batch shape: [{b},128,128,128]\n")

        except Exception as e:
            all_passed = False
            print(f"Hospital {h_id}:")
            print(f"    Patients: {patient_count}")
            print(f"    DataLoader: FAILED ({str(e)})\n")

    print(f"All hospital DataLoaders: {'PASSED' if all_passed else 'FAILED'}\n")
    print("=======================================================")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
