# It is important to use scaffold splitting see https://www.oloren.ai/blog/scaff_split.html
# Take inspiration from https://github.com/snap-stanford/ogb/
# See also: https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py#L1475
# See https://github.com/deepchem/deepchem/blob/master/deepchem/splits/tests/test_scaffold_splitter.py to use the scaffold splitter
# from deepchem.splits.splitters import ScaffoldSplitter

import logging
import random
from collections import defaultdict
from typing import List, Tuple

import datamol as dm
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from src.splitters import BaseSplitter

py_logger = logging.getLogger(__name__)


class ScaffoldSplitter(BaseSplitter):
    """Split the data into train, val and test sets using the scaffold
    splitting method."""

    def generate_scaffolds(self) -> List[List[int]]:
        """Generate the scaffolds.

        Inspired bypchem/splits/splitters.py#L1565
        """
        if self.compound_list is None:
            raise ValueError("Compound list is None.")

        scaffolds = defaultdict(list)

        py_logger.info("Generating scaffolds ...")
        for inchi in self.compound_list:
            scaffold = self._generate_scaffold(inchi, include_chirality=False)
            if scaffold is None:
                continue
            scaffolds[scaffold].append(inchi)

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

        py_logger.info("Generating sets ...")
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
    def _generate_scaffold(inchi: str, include_chirality: bool = False) -> str:
        mol = dm.from_inchi(inchi)

        if mol is None:
            py_logger.debug(f"Couldn't convert compound: {inchi}")
            return None

        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
