import torch
import torchvision.transforms.v2 as transforms

from src.modules.transforms import ImageNormalization


class DefaultJUMPTransform(torch.nn.Module):
    def __init__(self, size=256, dim=(-2, -1)):
        super().__init__()
        self.size = size
        self.dim = dim
        self.transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size),
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(),
            ImageNormalization(dim=dim),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.transform(inpt)
