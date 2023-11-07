# flake8: noqa: F401

import torchvision

from src.modules.transforms.default import ComplexTransform, ResizeTransform, SimpleTransform

torchvision.disable_beta_transforms_warning()
