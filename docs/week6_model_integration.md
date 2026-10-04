# Week 6 — 3D U-Net Model Integration

## Completed integration

- Flower server and hospital clients now create the same MONAI 3D U-Net
  (`channels=(4, 8, 16, 32)`, `strides=(2, 2, 2)`).
- The MRI data interface is `(N, 1, D, H, W)` for normalized volumes and
  `(N, D, H, W)` for integer voxel masks.
- `get_client_mri_dataloaders` is the hospital-side data entry point. It uses
  deterministic MRI-like development volumes until Sudheer's prepared MRI
  files are supplied; the tensor contract remains unchanged.
- The client records model parameter L1 norms immediately before and after its
  local update to prove that local training changed the shared model.

## Verified local hospital training

Command run with the Python 3.11 PyTorch/MONAI environment:

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/week6_local_training.py
```

| Check | Verified result |
| --- | --- |
| Automated tests | 6 passed |
| Device | CPU |
| Volume shape | `(1, 1, 16, 16, 16)` |
| Mask shape | `(1, 16, 16, 16)` |
| Hospital training/test cases | 6 / 2 |
| Local epochs | 2 |
| Training loss | 0.00028410 |
| Training voxel accuracy | 77.09% |
| Test loss | 0.00026857 |
| Test voxel accuracy | 81.96% |
| Parameter L1 before local fit | 2017.8035 |
| Parameter L1 after local fit | 2033.3593 |
| Parameter L1 change | +15.5558 |

The positive weight change confirms a local optimizer update occurred before
parameters are returned to Flower for federated aggregation. Re-run the script
to regenerate the machine-readable result at
`results/week6_local_training_results.json`.

## Three-hospital Flower simulation

The complete Flower simulation was verified with all three configured hospital
clients. The server and every client exited with status `0`.

| Round | Aggregated loss | Aggregated accuracy | Average client latency |
| --- | ---: | ---: | ---: |
| 1 | 0.00028605 | 80.44% | 2.788 s |
| 2 | 0.00023888 | 92.97% | 0.200 s |
| 3 | 0.00020252 | 97.93% | 0.190 s |

Run the reproducible Windows command below from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_simulation.ps1 -PythonPath "C:\Users\kunda\AppData\Local\Programs\Python\Python311\python.exe"
```

## Evidence screenshot

Use the terminal output from the two commands above as the Week 6 evidence
screenshot. It shows the passing tests, MRI tensor shapes, training metrics,
and weight change in one reproducible record.
