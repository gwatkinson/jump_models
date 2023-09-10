"""Module that takes fingerprints and passes them through a MLP."""

# flake8: noqa: B006

from typing import Callable, List, Optional, Union

import torch.nn as nn

from src.modules.layers.mlp import MLP
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class FingerprintsWithMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int = 512,
        embedding_dim: Union[int, List[int]] = [512, 512, 512],
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.backbone = MLP(
            input_dim,
            embedding_dim,
            embedding_dim=embedding_dim,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            dropout=dropout,
        )

        self.projection_head = nn.Linear(embedding_dim, out_dim)

        logger.info(f"Using Fingerprint MLP")

    def forward(self, x):
        z = self.backbone(x)
        z = self.projection_head(z)
        return z
