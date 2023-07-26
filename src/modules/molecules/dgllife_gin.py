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
        pretrained_name: str = "gin_supervised_masking",
        out_dim: int = 512,
        pooling: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.out_dim = out_dim
        self.pooling = pooling

        self.base_model = dgllife.model.load_pretrained(self.pretrained_name)
        self.pretrained_dim = self.base_model.node_embeddings[0].embedding_dim
        self.pooler = self.get_pooling(pooling)
        self.head = nn.Linear(self.pretrained_dim, self.out_dim)

        logger.info(f"Using pretrained model: {self.pretrained_name}")

    def extract(self, x):
        nfeats, efeats = self.get_nodes_edges_feats(x)
        node_feats = self.base_model(x, nfeats, efeats)
        z = self.pooler(x, node_feats)
        return z

    def forward(self, x):
        # x is a batch of DGLGraphs created in the custom collate_fn of the dataloader
        z = self.extract(x)
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
