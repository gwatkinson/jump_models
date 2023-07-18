# It is important to use scaffold splitting see https://www.oloren.ai/blog/scaff_split.html
# Take inspiration from https://github.com/snap-stanford/ogb/
# See also: https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py#L1475

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import datamol as dm
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import train_test_split

# See https://github.com/deepchem/deepchem/blob/master/deepchem/splits/tests/test_scaffold_splitter.py to use the scaffold splitter
# from deepchem.splits.splitters import ScaffoldSplitter


py_logger = logging.getLogger(__name__)


class BaseSplitter(ABC):
    """Base class for splitters."""

    def __init__(
        self,
        train: Union[float, int],
        val: Union[float, int],
        test: Union[float, int],
        compound_list: Optional[List[str]] = None,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.compound_list = compound_list
        self.normalized = False

        if self.compound_list is not None:
            py_logger.debug("Checking train, val and test values in init.")
            self.normalize_train_val_test()

    @property
    def input_type(self):
        if isinstance(self.train, int) and isinstance(self.val, int) and isinstance(self.test, int):
            return "int"
        elif isinstance(self.train, float) and isinstance(self.val, float) and isinstance(self.test, float):
            return "float"

    @property
    def n_compounds(self):
        """Return the number of compounds."""
        if self.compound_list is None:
            raise ValueError("Compound list is None.")
        return len(self.compound_list)

    def set_compound_list(self, compound_list: List[str]):
        """Set the compound list."""
        self.compound_list = compound_list
        py_logger.debug("Checking train, val and test values from set_compound_list function.")
        py_logger.debug(f"Before: Train: {self.train}, val: {self.val}, test: {self.test}")
        self.normalize_train_val_test()

    def normalize_train_val_test(self):
        """Normalize the train, val and test values."""
        total = self.train + self.val + self.test
        if self.input_type == "int":
            py_logger.debug("Train, val and test are integers.")
            if self.compound_list is not None and total > len(self.compound_list):
                py_logger.warning(
                    f"Total split size ({total}) is larger than the dataset size ({len(self.compound_list)})."
                )
                py_logger.warning("Normalizing train, val and test to 1.")
                self.train = self.train / total
                self.val = self.val / total
                self.test = self.test / total
                self.normalized = True
        elif self.input_type == "float":
            py_logger.debug("Train, val and test are floats. Normalizing to 1.")
            self.train = self.train / total
            self.val = self.val / total
            self.test = self.test / total
            self.normalized = True
        else:
            raise ValueError("Train, val and test must be either integers or floats.")

        py_logger.debug(f"Train: {self.train}, val: {self.val}, test: {self.test}")

    @abstractmethod
    def split(self) -> Tuple[List[str], List[str], List[str]]:
        """Split the data into train, val and test sets."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(train={self.train}, val={self.val}, test={self.test}, len_compound_list={len(self.compound_list)})"

    def __call__(self) -> Tuple[List[str], List[str], List[str]]:
        return self.split()


class RandomSplitter(BaseSplitter):
    """Split data randomly into train, val and test sets."""

    def split(self) -> Tuple[List[str], List[str], List[str]]:
        """Split the data into train, val and test sets."""
        if self.compound_list is None:
            raise ValueError("Compound list is None.")

        if self.input_type == "int":
            train_val, test = train_test_split(
                self.compound_list, test_size=self.test, train_size=self.train + self.val, random_state=42
            )
            train, val = train_test_split(train_val, test_size=self.val, train_size=self.train, random_state=42)
        elif self.normalized:
            train_val, test = train_test_split(self.compound_list, test_size=self.test, random_state=42)
            train, val = train_test_split(train_val, test_size=self.val / (self.train + self.val), random_state=42)
        else:
            raise ValueError("Error in the code.")

        return train, val, test


class ScaffoldSplitter(BaseSplitter):
    """Split the data into train, val and test sets using the scaffold
    splitting method."""

    def generate_scaffolds(self) -> List[List[int]]:
        """Generate the scaffolds.

        Inspired bypchem/splits/splitters.py#L1565
        """
        if self.compound_list is None:
            raise ValueError("Compound list is None.")

        scaffolds = {}

        py_logger.info("About to generate scaffolds")
        for smiles in self.compound_list:
            scaffold = self._generate_scaffold(smiles, include_chirality=False)
            if scaffold is None:
                continue
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [smiles]
            else:
                scaffolds[scaffold].append(smiles)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (_, scaffold_set) in sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        return scaffold_sets

    def split(self) -> Tuple[List[str], List[str], List[str]]:
        """Split the data into train, val and test sets."""
        if self.compound_list is None:
            raise ValueError("Compound list is None.")

        scaffold_sets = self.generate_scaffolds()
        random.shuffle(scaffold_sets)

        if self.input_type == "float":
            train_cutoff = int(self.train * self.n_compounds)
            val_cutoff = int(self.val * self.n_compounds)
            test_cutoff = int(self.test * self.n_compounds)
        elif self.input_type == "int":
            train_cutoff = self.train
            val_cutoff = self.val
            test_cutoff = self.test

        train_cpds: List[str] = []
        val_cpds: List[str] = []
        test_cpds: List[str] = []

        for scaffold_set in scaffold_sets:
            if len(train_cpds) + len(scaffold_set) <= train_cutoff:
                train_cpds += scaffold_set
            elif len(val_cpds) + len(scaffold_set) <= val_cutoff:
                val_cpds += scaffold_set
            elif len(test_cpds) + len(scaffold_set) <= test_cutoff:
                test_cpds += scaffold_set

        return train_cpds, val_cpds, test_cpds

    @staticmethod
    def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
        """Compute the Bemis-Murcko scaffold for a SMILES string.

        Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
        They are essentially that part of the molecule consisting of
        rings and the linker atoms between them.

        Paramters
        ---------
        smiles: str
            SMILES
        include_chirality: bool, default False
            Whether to include chirality in scaffolds or not.

        Returns
        -------
        str
            The MurckScaffold SMILES from the original SMILES

        References
        ----------
        .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
            1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.

        Note
        ----
        This function requires RDKit to be installed.
        """

        mol = dm.to_mol(smiles)

        if mol is None:
            py_logger.debug(f"Couldn't convert compound: {smiles}")
            return None

        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
