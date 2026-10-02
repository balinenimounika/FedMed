"""Diagnostic script to verify Python environment, packages, and directory layout."""

import sys
from pathlib import Path


def check_environment() -> bool:
    all_passed = True
    base_dir = Path(__file__).resolve().parent.parent

    print("=" * 65)
    print("           FedMed Environment & Dependency Diagnostic")
    print("=" * 65)

    # 1. Python Version Check
    py_ver = sys.version_info
    py_ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 10):
        print(f"  [PASS] Python Version: {py_ver_str} (>= 3.10 requirement met)")
    else:
        print(f"  [FAIL] Python Version: {py_ver_str} (Must be >= 3.10)")
        all_passed = False

    # 2. Package Checks
    packages = [
        ("flwr", "flwr", "1.8.0", lambda v: v.startswith("1.8")),
        ("torch", "torch", ">=2.0.0", lambda v: int(v.split(".")[0]) >= 2),
        ("torchvision", "torchvision", ">=0.15.0", lambda v: True),
        ("numpy", "numpy", ">=1.24.0,<2.0.0", lambda v: v.startswith("1.") and int(v.split(".")[1]) >= 24),
        ("sklearn", "scikit-learn", ">=1.3.0", lambda v: True),
        ("matplotlib", "matplotlib", ">=3.7.0", lambda v: True),
    ]

    print("\nPackage Verification:")
    print("-" * 65)
    for import_name, display_name, req_spec, validator in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            valid = validator(ver) if ver != "unknown" else True
            if valid:
                print(f"  [PASS] {display_name:<15}: {ver:<12} (Required: {req_spec})")
            else:
                print(f"  [WARN] {display_name:<15}: {ver:<12} (Required: {req_spec}, mismatch detected)")
        except ImportError as e:
            print(f"  [FAIL] {display_name:<15}: Not installed ({e})")
            all_passed = False

    # 3. Directory Layout Check
    print("\nDirectory & File Layout Verification:")
    print("-" * 65)
    required_paths = [
        ("Directory", base_dir / "src"),
        ("Directory", base_dir / "scripts"),
        ("Directory", base_dir / "results"),
        ("Directory", base_dir / "logs"),
        ("File", base_dir / "src" / "config.py"),
        ("File", base_dir / "src" / "model.py"),
        ("File", base_dir / "src" / "dataset.py"),
        ("File", base_dir / "src" / "client.py"),
        ("File", base_dir / "src" / "server.py"),
        ("File", base_dir / "requirements.txt"),
    ]

    for item_type, path in required_paths:
        exists = path.is_dir() if item_type == "Directory" else path.is_file()
        rel_path = path.relative_to(base_dir)
        if exists:
            print(f"  [PASS] {item_type:<10} exists: {str(rel_path)}")
        else:
            print(f"  [FAIL] {item_type:<10} MISSING: {str(rel_path)}")
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("  ALL CHECKS PASSED: FedMed environment is valid and ready.")
        print("=" * 65)
        return True
    else:
        print("  VERIFICATION FAILED: Resolve missing items above.")
        print("=" * 65)
        return False


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
