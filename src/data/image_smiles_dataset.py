"""Module containing a torch Dataset that returns a molecule and an associated
image."""

import logging
import random
from typing import Callable, Dict, List, Optional

import datamol as dm
import dgl
import pandas as pd
import torch
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, mol_to_bigraph
from torch.utils.data import Dataset

from src.data_utils.image_io import load_image_paths_to_array

py_logger = logging.getLogger(__name__)

default_channels = ["DNA", "AGP", "ER", "Mito", "RNA"]


class MoleculeImageDataset(Dataset):
    """This Dataset returns both a molecule, in the SMILES format and an
    associated image."""

    def __init__(
        self,
        load_df: pd.DataFrame,
        compound_dict: Dict[str, List[str]],
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        sampler: Optional[Callable[[List[str]], str]] = None,
        channels: List[str] = default_channels,
        col_fstring: str = "FileName_Orig{channel}",
    ):
        self.load_df = load_df
        self.compound_dict = compound_dict
        self.transform = transform
        self.compound_transform = compound_transform
        self.sampler = sampler or random.choice
        self.channels = channels
        self.col_fstring = col_fstring
        self.compound_list = list(self.compound_dict.keys())
        self.n_compounds = len(self.compound_list)
        self.image_list = self.load_df.index.tolist()
        self.n_images = len(self.image_list)

    def __len__(self):
        return self.n_compounds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={self.n_compounds}, n_images={self.n_images})"

    def __getitem__(self, idx):
        compound = self.compound_list[idx]  # An inchi or smiles string
        image_id = self.sampler(self.compound_dict[compound])  # An index into the load_df
        image_paths = [
            str(self.load_df.loc[image_id, self.col_fstring.format(channel=channel)]) for channel in self.channels
        ]

        img_array = load_image_paths_to_array(image_paths)  # A numpy array: (5, 768, 768)
        img_array = torch.from_numpy(img_array)

        if self.transform:
            img_array = self.transform(img_array)

        if self.compound_transform:
            compound = self.compound_transform(compound)

        return {"image": img_array, "compound": compound}


def image_graph_collate_function(data):
    """Collate function for the MoleculeImageDataset.

    Args:
        data: list of dicts with keys 'image' and 'compound'

    Returns:
        dict with keys 'image' and 'compound'
    """
    image = torch.stack([d["image"] for d in data])
    compound = dgl.batch([d["compound"] for d in data])

    return {"image": image, "compound": compound}


def graph_featurizer(mol):
    if isinstance(mol, str):
        mol = dm.to_mol(mol)

    g = mol_to_bigraph(
        mol,
        add_self_loop=True,
        node_featurizer=PretrainAtomFeaturizer(),
        edge_featurizer=PretrainBondFeaturizer(),
        canonical_atom_order=False,
    )

    return g


def molecule_transform(inchi):
    mol = dm.from_inchi(inchi)
    g = graph_featurizer(mol)

    return g
