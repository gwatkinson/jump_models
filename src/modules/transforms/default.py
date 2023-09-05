import torch
import torchvision.transforms.v2 as transforms

from src.modules.transforms.fill_nans import FillNaNs
from src.modules.transforms.image_normalization import ImageNormalization


class SimpleWithNormalize(torch.nn.Module):
    def __init__(self, size=256, dim=(-2, -1)):
        super().__init__()
        self.size = size
        self.dim = dim
        self.transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size, pad_if_needed=True),
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(),
            ImageNormalization(dim=dim),
            FillNaNs(nan=0.0, posinf=None, neginf=None),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.transform(inpt)


class NormalizeBeforeCrop(torch.nn.Module):
    def __init__(self, size=256, dim=(-2, -1)):
        super().__init__()
        self.size = size
        self.dim = dim
        self.transform = torch.nn.Sequential(
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ImageNormalization(dim=dim),
            transforms.RandomCrop(size, pad_if_needed=True),
            FillNaNs(nan=0.0, posinf=None, neginf=None),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.transform(inpt)


class SimpleTransform(torch.nn.Module):
    def __init__(self, size=256):
        super().__init__()
        self.size = size
        self.transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size, pad_if_needed=True),
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(),
            FillNaNs(nan=0.0, posinf=None, neginf=None),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.transform(inpt)
