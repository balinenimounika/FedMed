"""Configurable MONAI 3D U-Net for volumetric segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from monai.networks.nets import UNet


@dataclass(frozen=True)
class UNet3DConfig:
    """Architecture settings shared by every federated client.

    All clients must use exactly the same configuration so their model weights
    are compatible with federated averaging. ``out_channels`` includes the
    background class for multiclass segmentation.
    """

    in_channels: int = 1
    out_channels: int = 2
    channels: Sequence[int] = (16, 32, 64, 128, 256)
    strides: Sequence[int] = (2, 2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.0
    norm: str | tuple[str, dict] = "INSTANCE"
    act: str | tuple[str, dict] = ("PRELU", {"init": 0.2})
    bias: bool = True
    adn_ordering: str = "NDA"

    def validate(self) -> None:
        """Raise a useful error before MONAI constructs an invalid network."""
        if self.in_channels < 1:
            raise ValueError("in_channels must be at least 1")
        if self.out_channels < 1:
            raise ValueError("out_channels must be at least 1")
        if len(self.channels) < 2:
            raise ValueError("channels must contain at least two levels")
        if len(self.strides) != len(self.channels) - 1:
            raise ValueError("strides must be exactly one element shorter than channels")
        if any(channel < 1 for channel in self.channels):
            raise ValueError("all channel counts must be positive")
        if any(stride < 1 for stride in self.strides):
            raise ValueError("all strides must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")


def build_unet3d(config: UNet3DConfig | None = None) -> UNet:
    """Build the standard 3D U-Net used on each FedMed client.

    The returned module accepts tensors shaped ``(batch, channels, depth,
    height, width)`` and returns segmentation logits with the same spatial
    dimensions. Apply sigmoid or softmax only in the loss/metric pipeline.
    """
    config = config or UNet3DConfig()
    config.validate()
    model = UNet(
        spatial_dims=3,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=tuple(config.channels),
        strides=tuple(config.strides),
        num_res_units=config.num_res_units,
        dropout=config.dropout,
        norm=config.norm,
        act=config.act,
        bias=config.bias,
        adn_ordering=config.adn_ordering,
    )
    # Used by generic training utilities to select voxel-wise loss/metrics.
    model.spatial_dims = 3  # type: ignore[attr-defined]
    return model
