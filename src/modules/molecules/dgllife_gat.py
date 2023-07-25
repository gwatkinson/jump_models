"""Molecule encoder for the simple jump contrastive model."""

import logging

import datamol as dm
import dgllife
import torch.nn as nn
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph

logger = logging.getLogger(__name__)


def dgl_canonical_featurizer(
    inchi, add_self_loop=True, canonical_atom_order=True, num_virtual_nodes=0, explicit_hydrogens=False
):
    mol = dm.from_inchi(inchi)

    graph = mol_to_bigraph(
        mol,
        add_self_loop=add_self_loop,
        node_featurizer=CanonicalAtomFeaturizer(),
        edge_featurizer=CanonicalBondFeaturizer(self_loop=add_self_loop),
        canonical_atom_order=canonical_atom_order,
        num_virtual_nodes=num_virtual_nodes,
        explicit_hydrogens=explicit_hydrogens,
    )

    return graph


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

        self.base_model = dgllife.model.load_pretrained(self.pretrained_name)
        self.pretrained_dim = self.base_model.gnn.gnn_layers[0].gat_conv.fc.in_features
        self.head = nn.Linear(self.pretrained_dim, self.out_dim)

        logger.info(f"Using pretrained model: {self.pretrained_name}")

    def extract(self, x):
        node_feats = self.base_model(x, x.ndata["h"])
        return node_feats

    def forward(self, x):
        # x is a batch of DGLGraphs created in the custom collate_fn of the dataloader
        z = self.extract(x)
        z = self.head(z)
        return z
