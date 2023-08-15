"""Molecule encoder for the simple jump contrastive model."""

import dgllife
import torch.nn as nn

from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class GATPretrainedWithLinearHead(nn.Module):
    """A module that uses the pretrained GAT encoders from the dgllife library
    and adds a linear head on top."""

    def __init__(
        self,
        pretrained_name: str = "GAT_canonical_PCBA",
        out_dim: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.out_dim = out_dim

        self.backbone = dgllife.model.load_pretrained(self.pretrained_name)
        self.pretrained_dim = self.backbone.gnn.gnn_layers[0].gat_conv.fc.in_features
        self.projection_head = nn.Linear(self.pretrained_dim, self.out_dim)

        logger.info(f"Using pretrained model: {self.pretrained_name}")

    def extract(self, x):
        node_feats = self.backbone(x, x.ndata["h"])
        return node_feats

    def forward(self, x):
        # x is a batch of DGLGraphs created in the custom collate_fn of the dataloader
        z = self.extract(x)
        z = self.projection_head(z)
        return z
