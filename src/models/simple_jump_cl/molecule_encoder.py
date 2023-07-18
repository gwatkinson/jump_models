"""Molecule encoder for the simple jump contrastive model."""

import torch
import torch.nn as nn
from molfeat.trans.pretrained import PretrainedDGLTransformer


class GINPretrainedWithLinearHead(nn.Module):
    """A module that uses the pretrained GIN encoders from the molfeat library
    and adds a linear head on top."""

    def __init__(
        self,
        pretrained_name: str = "gin_supervised_infomax",
        out_dim: int = 2048,
        pretrained_dim: int = 300,
        **kwargs,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.pretrained_dim = pretrained_dim
        self.out_dim = out_dim

        self.pretrained = PretrainedDGLTransformer(kind=self.pretrained_name, dtype=torch.float32, **kwargs)
        self.head = nn.Linear(self.pretrained_dim, self.out_dim)

    def forward(self, x):
        x = self.pretrained(x)

        if self.device is None:
            self.device = x.device
            self.head = self.head.to(self.device)

        x = self.head(x)
        return x
