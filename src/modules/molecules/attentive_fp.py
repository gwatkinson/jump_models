"""Molecule encoder for the simple jump contrastive model."""

import torch.nn as nn
from dgllife.model import AttentiveFPPredictor

from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class AttentiveFPWithLinearHead(nn.Module):
    """A module that uses an AttentiveFP encoders from the dgllife library and
    adds a linear head on top."""

    def __init__(
        self,
        node_feat_size: int = 256,
        edge_feat_size: int = 128,
        num_layers: int = 4,
        num_timesteps: int = 2,
        graph_feat_size: int = 256,
        n_tasks: int = 256,
        dropout: float = 0.2,
        out_dim: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.graph_feat_size = graph_feat_size
        self.dropout = dropout
        self.out_dim = out_dim

        self.backbone = AttentiveFPPredictor(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=n_tasks,
            dropout=dropout,
        )

        self.pretrained_dim = n_tasks
        self.projection_head = nn.Linear(self.pretrained_dim, self.out_dim)

        logger.info("Using model: AttentiveFPWithLinearHead")

    def extract(self, x, get_node_weight=False):
        nfeats, efeats = self.get_nodes_edges_feats(x)
        return self.backbone(x, nfeats, efeats, get_node_weight=get_node_weight)

    def forward(self, x, get_node_weight=False):
        # x is a batch of DGLGraphs created in the custom collate_fn of the dataloader
        if get_node_weight:
            z, node_weight = self.extract(x, get_node_weight=get_node_weight)
            z = self.projection_head(z)
            return z, node_weight
        else:
            z = self.extract(x, get_node_weight=get_node_weight)
            z = self.projection_head(z)
            return z

    @staticmethod
    def get_nodes_edges_feats(x):
        nfeats = x.ndata["h"]
        efeats = x.edata["e"]

        return nfeats, efeats
