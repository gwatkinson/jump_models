from abc import ABC, abstractmethod
from typing import Literal

import datamol as dm

from src.modules.collate_fn import default_collate


class DefaultCompoundTransform(ABC):
    def __init__(
        self,
        compound_str_type: Literal["inchi", "smiles", "selfies", "smarts"] = "smiles",
    ):
        self.compound_str_type = compound_str_type

    def convert_str_to_mol(self, compound_str: str):
        if self.compound_str_type == "inchi":
            mol = dm.from_inchi(compound_str)
        elif self.compound_str_type == "smiles":
            mol = dm.to_mol(compound_str)
        elif self.compound_str_type == "selfies":
            mol = dm.from_selfies(compound_str)
        elif self.compound_str_type == "smarts":
            mol = dm.from_smarts(compound_str)
        else:
            raise ValueError(f"Unknown compound_str_type: {self.compound_str_type}")

        return dm.to_smiles(mol)

    @abstractmethod
    def mol_to_feat(self, mol: str):
        raise NotImplementedError

    def __call__(self, compound_str: str):
        mol = self.convert_str_to_mol(compound_str)
        feats = self.mol_to_feat(mol)
        return feats

    def get_default_collate_fn(self):
        return default_collate
