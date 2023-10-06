import torch
import torchvision.transforms as T

from src.modules.transforms.color_jitter import ColorJitterPerChannel
from src.modules.transforms.drop_channel import DropTransform
from src.modules.transforms.fill_nans import FillNaNs
from src.modules.transforms.image_normalization import ImageNormalization


class SimpleWithNormalize(torch.nn.Module):
    def __init__(self, size=256, dim=(-2, -1)):
        super().__init__()
        self.size = size
        self.dim = dim
        self.transform = torch.nn.Sequential(
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomCrop(size, pad_if_needed=True),
            T.ToImageTensor(),
            T.ConvertImageDtype(),
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
            T.ToImageTensor(),
            T.ConvertImageDtype(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            ImageNormalization(dim=dim),
            T.RandomCrop(size, pad_if_needed=True),
            FillNaNs(nan=0.0, posinf=None, neginf=None),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.transform(inpt)


class SimpleTransform(T.Compose):
    def __init__(self, size=256, p=0.5):
        super().__init__(
            (
                T.Lambda(lambda x: x.transpose(1, 2, 0) if x.shape[0] == 5 else x),
                T.ToTensor(),
                T.RandomHorizontalFlip(p=p),
                T.RandomVerticalFlip(p=p),
                T.RandomCrop(size, pad_if_needed=True),
                FillNaNs(nan=0.0, posinf=None, neginf=None),
            )
        )


class ComplexTransform(T.Compose):
    def __init__(
        self,
        size=512,
        flip_p=0.3,
        drop_p=0.3,
        resize_p=0.3,
        gaussian_p=0.8,
        color_p=0.8,
        resize_min_ratio=0.9,
        kernel_size=23,
        sigma=(1.0, 3.0),
        intensity=0.3,
        brightness=0.5,
        use_flip=False,
        use_blur=False,
        use_color_jitter=False,
        use_drop=False,
        use_resized_crop=False,
        fill_nan=False,
    ):
        sigma = (float(sigma[0]), float(sigma[1]))
        transforms = [
            T.Lambda(lambda x: x.transpose(1, 2, 0) if x.shape[0] == 5 else x),
            T.ToTensor(),
        ]

        if use_flip:
            transforms.append(T.RandomHorizontalFlip(p=flip_p))
            transforms.append(T.RandomVerticalFlip(p=flip_p))

        sub_transforms = []
        if use_blur:
            sub_transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=kernel_size, sigma=sigma), FillNaNs(nan=0.0, posinf=None, neginf=None)],
                    p=gaussian_p,
                )
            )

        if use_color_jitter:
            sub_transforms.append(
                T.RandomApply([ColorJitterPerChannel(intensity=intensity, brightness=brightness)], p=color_p)
            )

        if len(sub_transforms) > 0:
            transforms.append(T.RandomOrder(sub_transforms))

        if use_resized_crop:
            transforms.append(
                T.RandomChoice(
                    [
                        T.RandomResizedCrop(
                            size=size,
                            scale=(resize_min_ratio * size / 768, 1.0),
                            ratio=(1.0, 1.0),
                            interpolation=2,
                            antialias=True,
                        ),
                        T.RandomCrop(size, pad_if_needed=True),
                    ],
                    p=[resize_p, 1.0 - resize_p],
                )
            )
        else:
            transforms.append(T.RandomCrop(size, pad_if_needed=False))

        if use_drop:
            transforms.append(DropTransform(p=drop_p))

        if fill_nan:
            transforms.append(FillNaNs(nan=0.0, posinf=None, neginf=None))

        super().__init__(transforms)
