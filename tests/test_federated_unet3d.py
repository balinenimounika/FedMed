import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from src.config import MRI_BATCH_SIZE, MRI_IN_CHANNELS, MRI_OUT_CHANNELS, MRI_UNET_CHANNELS, MRI_UNET_STRIDES
from src.dataset import get_client_mri_dataloaders
from src.fedmed.models import UNet3DConfig, build_unet3d
from src.model import get_parameters, parameter_l1_norm, set_parameters, train


def test_mri_loader_matches_unet_input_contract() -> None:
    train_loader, _ = get_client_mri_dataloaders(0, batch_size=MRI_BATCH_SIZE)
    volume, mask = next(iter(train_loader))
    assert volume.ndim == 5 and volume.shape[1] == 1
    assert mask.shape == volume.shape[:1] + volume.shape[2:]


def test_local_training_changes_unet_weights() -> None:
    torch.manual_seed(42)
    loader, _ = get_client_mri_dataloaders(0, batch_size=1, total_samples=4)
    model = build_unet3d(UNet3DConfig(in_channels=MRI_IN_CHANNELS, out_channels=MRI_OUT_CHANNELS, channels=MRI_UNET_CHANNELS, strides=MRI_UNET_STRIDES))
    before = parameter_l1_norm(model)
    loss, _ = train(model, loader, epochs=1, verbose=False)
    after = parameter_l1_norm(model)
    assert loss > 0
    assert after != before


def test_unet_parameters_round_trip_for_federated_exchange() -> None:
    config = UNet3DConfig(in_channels=MRI_IN_CHANNELS, out_channels=MRI_OUT_CHANNELS, channels=MRI_UNET_CHANNELS, strides=MRI_UNET_STRIDES)
    source, target = build_unet3d(config), build_unet3d(config)
    parameters = get_parameters(source)
    set_parameters(target, parameters)
    assert parameter_l1_norm(source) == parameter_l1_norm(target)
