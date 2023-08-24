# flake8: noqa: F401

import torchvision

from src.modules.transforms.default import DefaultJUMPTransform

torchvision.disable_beta_transforms_warning()
