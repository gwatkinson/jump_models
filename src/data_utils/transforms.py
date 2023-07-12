"""Contains custom transforms and default Compose for easy access."""

import torch
import torchvision.transforms.v2 as transforms


class ImageNormalization(torch.nn.Module):
    """Self normalize the images by their mean and std by channel.

    The input format should be (batch, channels, height, width). The
    output format will be the same.
    """

    def __init__(self, dim=(2, 3)):
        super().__init__()
        self.dim = dim

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        mean = inpt.mean(dim=self.dim, keepdim=True)
        std = inpt.std(dim=self.dim, keepdim=True)

        return (inpt - mean) / std


class DefaultJUMPTransform(torch.nn.Module):
    def __init__(self, size=256, dim=(-2, -1)):
        super().__init__()
        self.size = size
        self.dim = dim
        self.transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            ImageNormalization(dim=dim),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.transform(inpt)
