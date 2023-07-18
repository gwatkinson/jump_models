"""Molecule encoder for the simple jump contrastive model."""


import dgllife
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, SMILESToBigraph


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

        self.smiles_to_graph = SMILESToBigraph(
            add_self_loop=False,
            node_featurizer=PretrainAtomFeaturizer(),
            edge_featurizer=PretrainBondFeaturizer(),
            canonical_atom_order=True,
            explicit_hydrogens=False,
            num_virtual_nodes=0,
        )

        self.model = dgllife.model.load_pretrained(self.pretrained_name)
        self.pooler = self.get_pooling(pooling)

        self.head = nn.Linear(self.pretrained_dim, self.out_dim)

    def forward(self, x):
        x = self.smiles_to_graph(x)
        x = self.model(x)
        x = self.pooler(x)
        x = self.head(x)
        return x

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
