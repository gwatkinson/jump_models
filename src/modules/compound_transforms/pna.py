# From https://github.com/HannesStark/3DInfomax/blob/master/commons/mol_encoder.py

from typing import Literal

import datamol as dm
import dgl
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

# from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from src.modules.compound_transforms.base_compound_transform import DefaultCompoundTransform
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class PNATransform(DefaultCompoundTransform):
    def __init__(
        self,
        compound_str_type: Literal["inchi", "smiles", "selfies", "smarts"] = "smiles",
    ):
        super().__init__(compound_str_type)

    def mol_to_feat(self, smiles: str) -> torch.Tensor:
        mol = dm.to_mol(smiles)
        n_atoms = len(mol.GetAtoms())

        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))

        atom_features = torch.tensor(atom_features_list, dtype=torch.long)

        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        if len(edges_list) == 0:
            # if n_atoms == 2:
            #     edges_list = [(0, 1), (1, 0)]
            #     edge_features_list = [[0, 0, 0], [0, 0, 0]]
            py_logger.warning(f"Empty edges for {smiles}")
            return None

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(edges_list, dtype=torch.long).T
        edge_features = torch.tensor(edge_features_list, dtype=torch.long)

        graph = dgl.graph(
            data=(edge_index[0], edge_index[1]),
            num_nodes=n_atoms,
        )

        graph.ndata["feat"] = atom_features
        graph.edata["feat"] = edge_features

        return graph
