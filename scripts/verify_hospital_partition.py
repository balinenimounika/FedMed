"""
Script: Read-Only 3-Hospital Partition Verification
---------------------------------------------------
Verifies the partition manifests:
  - data/hospitals/hospital_1.json
  - data/hospitals/hospital_2.json
  - data/hospitals/hospital_3.json

Checks counts, intersections, missing patients, and duplicates.
DOES NOT MODIFY ANY FILES.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main():
    hospitals_dir = Path("data/hospitals")
    processed_dir = Path("data/processed")

    h1_file = hospitals_dir / "hospital_1.json"
    h2_file = hospitals_dir / "hospital_2.json"
    h3_file = hospitals_dir / "hospital_3.json"

    for f in [h1_file, h2_file, h3_file]:
        if not f.exists():
            print(f"[ERROR] Manifest not found: {f}")
            sys.exit(1)

    with open(h1_file, "r", encoding="utf-8") as fp:
        h1_patients = json.load(fp).get("patient_ids", [])
    with open(h2_file, "r", encoding="utf-8") as fp:
        h2_patients = json.load(fp).get("patient_ids", [])
    with open(h3_file, "r", encoding="utf-8") as fp:
        h3_patients = json.load(fp).get("patient_ids", [])

    h1_set = set(h1_patients)
    h2_set = set(h2_patients)
    h3_set = set(h3_patients)

    all_processed = {f.stem for f in processed_dir.glob("*.npz")}

    h1_h2_overlap = h1_set.intersection(h2_set)
    h1_h3_overlap = h1_set.intersection(h3_set)
    h2_h3_overlap = h2_set.intersection(h3_set)

    total_assignments = len(h1_patients) + len(h2_patients) + len(h3_patients)
    unique_patients = h1_set | h2_set | h3_set

    missing_patients = all_processed - unique_patients
    duplicated_patients = h1_h2_overlap | h1_h3_overlap | h2_h3_overlap

    is_passed = (
        len(h1_patients) == 2 and
        len(h2_patients) == 2 and
        len(h3_patients) == 2 and
        total_assignments == 6 and
        len(unique_patients) == 6 and
        len(duplicated_patients) == 0 and
        len(missing_patients) == 0
    )

    print("=======================================================")
    print("3-HOSPITAL PARTITION VERIFICATION")
    print("=======================================================\n")

    print(f"Hospital 1 patients: {len(h1_patients)}")
    print(f"Hospital 2 patients: {len(h2_patients)}")
    print(f"Hospital 3 patients: {len(h3_patients)}\n")

    print(f"Total patient assignments: {total_assignments}")
    print(f"Unique patients: {len(unique_patients)}\n")

    print(f"Hospital 1 ∩ Hospital 2: {h1_h2_overlap}")
    print(f"Hospital 1 ∩ Hospital 3: {h1_h3_overlap}")
    print(f"Hospital 2 ∩ Hospital 3: {h2_h3_overlap}\n")

    print(f"Missing patients: {missing_patients}")
    print(f"Duplicated patients: {duplicated_patients}\n")

    print(f"Patient leakage: {'NONE' if not duplicated_patients else 'DETECTED'}")
    print(f"Hospital partition: {'PASSED' if is_passed else 'FAILED'}\n")
    print("=======================================================")

if __name__ == "__main__":
    main()
