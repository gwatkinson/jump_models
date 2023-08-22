# flake8: noqa: F401

import torchvision

from src.modules.transforms.default import DefaultJUMPTransform
from src.modules.transforms.image_normalization import ImageNormalization

torchvision.disable_beta_transforms_warning()
