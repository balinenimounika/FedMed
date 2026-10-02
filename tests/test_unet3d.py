import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from fedmed.models import UNet3DConfig, build_unet3d


def test_default_unet_preserves_spatial_shape() -> None:
    model = build_unet3d()
    volume = torch.randn(1, 1, 32, 32, 32)
    with torch.inference_mode():
        logits = model(volume)
    assert logits.shape == (1, 2, 32, 32, 32)


def test_unet_supports_multimodal_multiclass_configuration() -> None:
    config = UNet3DConfig(in_channels=4, out_channels=3, channels=(8, 16, 32), strides=(2, 2))
    model = build_unet3d(config)
    with torch.inference_mode():
        logits = model(torch.randn(1, 4, 16, 16, 16))
    assert logits.shape == (1, 3, 16, 16, 16)


def test_invalid_stride_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="one element shorter"):
        build_unet3d(UNet3DConfig(channels=(16, 32, 64), strides=(2,)))
