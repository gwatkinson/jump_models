import torch


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
