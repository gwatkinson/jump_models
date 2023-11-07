# It is important to use scaffold splitting see https://www.oloren.ai/blog/scaff_split.html
# Take inspiration from https://github.com/snap-stanford/ogb/
# See also: https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py#L1475
# See https://github.com/deepchem/deepchem/blob/master/deepchem/splits/tests/test_scaffold_splitter.py to use the scaffold splitter
# from deepchem.splits.splitters import ScaffoldSplitter

import random
from collections import defaultdict
from typing import Dict, List

import datamol as dm
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from src.splitters import BaseSplitter
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class ScaffoldSplitter(BaseSplitter):
    """Split the data into train, val and test sets using the scaffold
    splitting method."""

    def split(self) -> Dict[str, List[str]]:
        """Split the data into train, val and test sets."""
        if self.compound_list is None:
            raise ValueError("Compound list is None.")

        scaffold_sets = self.generate_scaffolds()
        random.shuffle(scaffold_sets)

        py_logger.info("Generating sets ...")

        val_cutoff = self.val
        test_cutoff = self.test
        retrieval_cutoff = self.retrieval

        total_train_cpds: List[str] = []
        val_cpds: List[str] = []
        test_cpds: List[str] = []
        retrieval_cpds: List[str] = []

        for scaffold_set in scaffold_sets:
            if len(val_cpds) + len(scaffold_set) <= val_cutoff:
                val_cpds += scaffold_set
            elif len(test_cpds) + len(scaffold_set) <= test_cutoff:
                test_cpds += scaffold_set
            elif len(retrieval_cpds) + len(scaffold_set) <= retrieval_cutoff:
                retrieval_cpds += scaffold_set
            else:
                total_train_cpds += scaffold_set  # Put all the other compounds in the train set

        random.shuffle(val_cpds)
        random.shuffle(test_cpds)
        random.shuffle(total_train_cpds)

        output = {
            "total_train": total_train_cpds,
            "val": val_cpds,
            "test": test_cpds,
        }

        if retrieval_cutoff > 0:
            output["retrieval"] = retrieval_cpds

        self.total_train_cpds = total_train_cpds
        self.val_cpds = val_cpds
        self.test_cpds = test_cpds
        self.retrieval_cpds = retrieval_cpds

        return output

    def split_train(self, total_train_cpds=None) -> List[str]:
        total_train_cpds = total_train_cpds or self.total_train_cpds
        train_cutoff = self.train
        scaffold_sets = self.generate_scaffolds(total_train_cpds)

        train_cpds = []

        for scaffold_set in scaffold_sets:
            if len(train_cpds) + len(scaffold_set) <= train_cutoff:
                train_cpds += scaffold_set
            elif len(train_cpds) == train_cutoff:
                break

        random.shuffle(train_cpds)

        self.train_cpds = train_cpds

        return train_cpds

    def generate_scaffolds(self, compound_list=None) -> List[List[int]]:
        """Generate the scaffolds.

        Inspired bypchem/splits/splitters.py#L1565
        """
        compound_list = compound_list or self.compound_list

        scaffolds = defaultdict(list)

        py_logger.info("Generating scaffolds ...")
        for inchi in compound_list:
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

    @staticmethod
    def _generate_scaffold(inchi: str, include_chirality: bool = False) -> str:
        mol = dm.from_inchi(inchi)

        if mol is None:
            py_logger.debug(f"Couldn't convert compound: {inchi}")
            return None

        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
