"""Molecule encoder for the simple jump contrastive model."""

import logging

import dgllife
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling

logger = logging.getLogger(__name__)


class GINPretrainedWithLinearHead(nn.Module):
    """A module that uses the pretrained GIN encoders from the molfeat library
    and adds a linear head on top."""

    def __init__(
        self,
        pretrained_name: str = "gin_supervised_infomax",
        out_dim: int = 2048,
        pretrained_dim: int = 300,
        pooling: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.pretrained_dim = pretrained_dim
        self.out_dim = out_dim
        self.pooling = pooling

        self.base_model = dgllife.model.load_pretrained(self.pretrained_name)
        self.pooler = self.get_pooling(pooling)
        self.head = nn.Linear(self.pretrained_dim, self.out_dim)

        logger.info(f"Using pretrained model: {self.pretrained_name}")
        logger.info(f"On device: {next(self.parameters()).device}")

    def forward(self, x):
        # x is a batch of DGLGraphs created in the custom collate_fn of the dataloader
        nfeats, efeats = self.get_nodes_edges_feats(x)

        try:
            logger.debug(f"Node feats device: {nfeats[0].device}")
            logger.debug(f"Edge feats device: {efeats[0].device}")
        except Exception:
            logger.warning("Could not get device for node and edge features")

        node_feats = self.base_model(x, nfeats, efeats)
        z = self.pooler(x, node_feats)
        z = self.head(z)
        return z

    @staticmethod
    def get_pooling(pooling: str):
        """Get pooling method from name.

        Args:
            pooling: name of the pooling method
        """
        pooling = pooling.lower()
        if pooling in ["mean", "avg", "average"]:
            return AvgPooling()
        elif pooling == "sum":
            return SumPooling()
        elif pooling == "max":
            return MaxPooling()
        else:
            raise ValueError(f"Pooling: {pooling} not supported !")

    @staticmethod
    def get_nodes_edges_feats(bg):
        nfeats = [bg.ndata[k] for k in bg.ndata]
        efeats = [bg.edata[k] for k in bg.edata]

        return nfeats, efeats
