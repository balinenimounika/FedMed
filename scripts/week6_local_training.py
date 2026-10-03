"""Run a deterministic hospital-side 3D U-Net training check and record results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOCAL_EPOCHS, MRI_BATCH_SIZE, MRI_TRAINING_RESULTS_PATH, RANDOM_SEED
from src.dataset import get_client_mri_dataloaders
from src.fedmed.models import UNet3DConfig, build_unet3d
from src.config import MRI_IN_CHANNELS, MRI_OUT_CHANNELS, MRI_UNET_CHANNELS, MRI_UNET_STRIDES
from src.model import parameter_l1_norm, test, train


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")
    train_loader, test_loader = get_client_mri_dataloaders(0, batch_size=MRI_BATCH_SIZE)
    volume, mask = next(iter(train_loader))
    model = build_unet3d(UNet3DConfig(
        in_channels=MRI_IN_CHANNELS, out_channels=MRI_OUT_CHANNELS,
        channels=MRI_UNET_CHANNELS, strides=MRI_UNET_STRIDES,
    )).to(device)
    before = parameter_l1_norm(model)
    train_loss, voxel_accuracy = train(model, train_loader, LOCAL_EPOCHS, device, verbose=False)
    after = parameter_l1_norm(model)
    test_loss, test_voxel_accuracy = test(model, test_loader, device)
    result = {
        "task": "week_6_local_hospital_3d_unet_check",
        "device": str(device),
        "input_shape": list(volume.shape),
        "mask_shape": list(mask.shape),
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "epochs": LOCAL_EPOCHS,
        "train_loss": train_loss,
        "train_voxel_accuracy": voxel_accuracy,
        "test_loss": test_loss,
        "test_voxel_accuracy": test_voxel_accuracy,
        "weight_l1_before": before,
        "weight_l1_after": after,
        "weight_l1_change": after - before,
    }
    MRI_TRAINING_RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
