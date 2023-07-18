"""Molecule encoder for the simple jump contrastive model."""

import logging
from typing import List

import datamol as dm
import dgl
import dgllife
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, mol_to_bigraph

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

        self.model = dgllife.model.load_pretrained(self.pretrained_name)
        self.pooler = self.get_pooling(pooling)
        self.head = nn.Linear(self.pretrained_dim, self.out_dim)

    def forward(self, x):
        bg = self.graph_featurizer(x)
        nfeats, efeats = self.get_n_e_feats(bg)
        node_feats = self.model(bg, nfeats, efeats)
        z = self.pooler(bg, node_feats)
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
    def graph_featurizer(smiles: List[str]):
        graphs = []
        success = []
        for smi in smiles:
            try:
                mol = dm.to_mol(smi)
                if mol is None:
                    success.append(False)
                    continue
                g = mol_to_bigraph(
                    mol,
                    add_self_loop=True,
                    node_featurizer=PretrainAtomFeaturizer(),
                    edge_featurizer=PretrainBondFeaturizer(),
                    canonical_atom_order=False,
                )
                graphs.append(g)
                success.append(True)
            except Exception as e:
                logger.error(e)
                success.append(False)
        return dgl.batch(graphs)

    @staticmethod
    def get_n_e_feats(bg):
        nfeats = [bg.ndata[k] for k in bg.ndata]
        efeats = [bg.edata[k] for k in bg.edata]

        return nfeats, efeats
