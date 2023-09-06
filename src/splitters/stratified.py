from typing import Dict, List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split

from src.splitters import BaseSplitter
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class StratifiedSplitter(BaseSplitter):
    def __init__(
        self,
        train: float,
        val: float,
        test: float,
        compound_list: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.train = train
        self.val = val
        self.test = test

        self.random_state = random_state

        self.compound_list = compound_list

    @property
    def n_compounds(self):
        """Return the number of compounds."""
        if self.compound_list is None:
            raise ValueError("Compound list is None.")
        return len(self.compound_list)

    def set_compound_list(self, compound_list: List[str]):
        """Set the compound list."""
        self.compound_list = compound_list
        self.normalize_train_val_test()

    def __call__(
        self,
    ) -> Union[Tuple[List[str], List[str], List[str]], Tuple[List[str], List[str], List[str], List[str]]]:
        return self.split()

    def split(self) -> Dict[str, List[str]]:
        """Split the data into train, val and test sets."""
        if self.compound_list is None:
            raise ValueError("Label list is None.")

        ids = list(range(len(self.compound_list)))

        train_val, test = train_test_split(
            ids, test_size=self.test, random_state=self.random_state, stratify=self.compound_list
        )
        sub_labels = [self.compound_list[i] for i in train_val]
        train, val = train_test_split(
            train_val,
            test_size=self.val / (self.train + self.val),
            random_state=self.random_state,
            stratify=sub_labels,
        )

        return {"train": train, "val": val, "test": test}

    def split_train(self) -> List[str]:
        """Subsplit a train list into a smaller train list."""
        raise NotImplementedError
