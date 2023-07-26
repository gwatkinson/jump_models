import datamol as dm
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    MolToBigraph,
    PretrainAtomFeaturizer,
    PretrainBondFeaturizer,
)


class DGLTransform:
    """Base class for a compound transform that converts a compound string into
    a DGLGraph."""

    compound_str_type = None
    af = None
    bf = None

    def __init__(
        self,
        add_self_loop: bool = True,
        canonical_atom_order: bool = True,
        num_virtual_nodes: int = 0,
        explicit_hydrogens: bool = False,
    ):
        self.compound_str_type = self.compound_str_type
        self.atom_featurizer = self.af()
        self.bond_featurizer = self.bf(self_loop=add_self_loop)
        self.add_self_loop = add_self_loop
        self.canonical_atom_order = canonical_atom_order
        self.num_virtual_nodes = num_virtual_nodes
        self.explicit_hydrogens = explicit_hydrogens

        self.mol_to_graph = MolToBigraph(
            add_self_loop=self.add_self_loop,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
            canonical_atom_order=self.canonical_atom_order,
            num_virtual_nodes=self.num_virtual_nodes,
            explicit_hydrogens=self.explicit_hydrogens,
        )

    def convert_str_to_mol(self, compound_str: str):
        if self.compound_str_type == "inchi":
            return dm.from_inchi(compound_str)
        elif self.compound_str_type == "smiles":
            return dm.to_mol(compound_str)
        elif self.compound_str_type == "selfies":
            return dm.from_selfies(compound_str)
        elif self.compound_str_type == "smarts":
            return dm.from_smarts(compound_str)

        raise ValueError(f"Unknown compound_str_type: {self.compound_str_type}")

    def __call__(self, compound_str: str):
        mol = self.convert_str_to_mol(compound_str)
        graph = self.mol_to_graph(mol)
        return graph


class DGLCanonicalFromInchi(DGLTransform):
    """Converts an inchi string into a DGLGraph using the
    CanonicalAtomFeaturizer and CanonicalBondFeaturizer."""

    compound_str_type = "inchi"
    af = CanonicalAtomFeaturizer
    bf = CanonicalBondFeaturizer


class DGLCanonicalFromSmiles(DGLTransform):
    """Converts a smiles string into a DGLGraph using the
    CanonicalAtomFeaturizer and CanonicalBondFeaturizer."""

    compound_str_type = "smiles"
    af = CanonicalAtomFeaturizer
    bf = CanonicalBondFeaturizer


class DGLPretrainedFromInchi(DGLTransform):
    """Converts an inchi string into a DGLGraph using the
    PretrainAtomFeaturizer and PretrainBondFeaturizer."""

    compound_str_type = "inchi"
    af = PretrainAtomFeaturizer
    bf = PretrainBondFeaturizer


class DGLPretrainedFromSmiles(DGLTransform):
    """Converts a smiles string into a DGLGraph using the
    PretrainAtomFeaturizer and PretrainBondFeaturizer."""

    compound_str_type = "smiles"
    af = PretrainAtomFeaturizer
    bf = PretrainBondFeaturizer
