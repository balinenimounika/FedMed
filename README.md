# FedMed

Federated learning baseline for 3D medical-image segmentation.

## Model setup (Kundan)

The project includes a configurable [MONAI](https://monai.io/) 3D U-Net in
`src/fedmed/models/unet3d.py`. Its default configuration targets a single
imaging modality and binary segmentation: one input channel and two output
logit channels (background plus foreground).

Install the baseline and run its checks:

```powershell
python -m pip install -e ".[dev]"
pytest
```

Use the shared factory on every Flower client so parameter shapes remain
identical for federated averaging:

```python
from fedmed.models import UNet3DConfig, build_unet3d

model = build_unet3d(UNet3DConfig(in_channels=1, out_channels=2))
```

## Team responsibilities

- Mehaboob: integration, testing, and documentation
- Mounika: Flower server/client setup
- Sudheer: dataset collection and preprocessing
- Kundan: PyTorch/MONAI 3D U-Net setup
