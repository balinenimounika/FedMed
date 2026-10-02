"""
Script: Partition Dataset Across 3 Simulated Hospitals
------------------------------------------------------
Deterministically partitions the 6 preprocessed BraTS patient scans across 3 hospitals:
  - Hospital 1: BraTS20_Training_001, BraTS20_Training_002
  - Hospital 2: BraTS20_Training_003, BraTS20_Training_004
  - Hospital 3: BraTS20_Training_005, BraTS20_Training_006
Guarantees patient-level isolation (zero slice or patient leakage).
Saves hospital manifests and distribution statistics.
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

from configs.dataset_config import DEFAULT_CONFIG
from src.data.hospital_partitioner import HospitalPartitioner


def main():
    print("=======================================================")
    print(" FedMed: Partitioning Dataset Across 3 Hospitals")
    print("=======================================================\n")

    partitioner = HospitalPartitioner(DEFAULT_CONFIG)

    # Step 1: Create partitions
    splits = partitioner.partition_patients()
    for h_key, pids in splits.items():
        h_id = h_key.split('_')[1]
        print(f"Hospital {h_id}:")
        for pid in pids:
            print(f"    {pid}")
        print()

    # Step 2: Compute distribution statistics
    print("[Stats] Computing hospital data distribution metrics...")
    stats = partitioner.compute_distribution_statistics()
    print(f"  -> Generated data/hospitals/hospital_statistics.json")
    print(f"  -> Generated reports/week1_hospital_distribution.txt")

    print("\n[SUCCESS] 3-Hospital partitioning complete.\n")


if __name__ == "__main__":
    main()
