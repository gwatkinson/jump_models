from typing import Any, Dict, List, Literal, Optional

import datamol as dm
from molfeat.trans.concat import FeatConcat


class FPTransform:
    def __init__(
        self,
        fps: Optional[List[str]] = None,
        compound_str_type: Literal["inchi", "smiles", "selfies", "smarts"] = "smiles",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.fps = fps or ["maccs", "ecfp"]
        self.compound_str_type = compound_str_type
        self.params = params or ({"ecfp": {"radius": 2}} if "ecfp" in self.fps else {})

        self.mol_to_feat = FeatConcat(fps, params=params)

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

        return mol

    def __call__(self, compound_str: str):
        mol = self.convert_str_to_mol(compound_str)
        feats = self.mol_to_feat(mol)
        return feats
