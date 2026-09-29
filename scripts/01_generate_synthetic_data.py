"""
Script 01: Generate Synthetic BraTS Dataset
-------------------------------------------
Generates synthetic 3D BraTS-compatible patient scans (FLAIR, T1, T1ce, T2, and segmentation mask)
in data/raw/ for out-of-the-box validation and testing.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.dataset_config import DEFAULT_CONFIG
from src.data.synthetic_sample import generate_synthetic_patient


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 3D Brain MRI scans for testing.")
    parser.add_argument("--num_patients", type=int, default=6, help="Number of synthetic patients to generate (default: 6)")
    parser.add_argument("--depth", type=int, default=155, help="Z slices (BraTS default: 155)")
    parser.add_argument("--height", type=int, default=240, help="Y height (BraTS default: 240)")
    parser.add_argument("--width", type=int, default=240, help="X width (BraTS default: 240)")
    args = parser.parse_args()

    raw_dir = Path(DEFAULT_CONFIG.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" FedMed: Generating {args.num_patients} Synthetic BraTS Scans")
    print(f" Destination: {raw_dir}")
    print(f" Volume Shape: ({args.depth}, {args.height}, {args.width})")
    print(f"=======================================================\n")

    for i in range(1, args.num_patients + 1):
        pid = f"BraTS20_Training_{i:03d}"
        seed = 100 + i
        print(f"Generating patient [{i}/{args.num_patients}]: {pid} ...")
        patient_path = generate_synthetic_patient(
            patient_id=pid,
            output_dir=raw_dir,
            shape=(args.depth, args.height, args.width),
            seed=seed
        )
        print(f"  -> Saved files in: {patient_path}")

    print(f"\n[SUCCESS] Generated {args.num_patients} synthetic patient records in {raw_dir}.\n")


if __name__ == "__main__":
    main()
