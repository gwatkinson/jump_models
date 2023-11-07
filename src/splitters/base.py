from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class BaseSplitter(ABC):
    def __init__(
        self,
        train: int,
        val: int,
        test: int,
        retrieval: int = 0,
        compound_list: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.retrieval = retrieval or 0

        self.random_state = random_state

        self.compound_list = compound_list
        self.normalized = False

        if self.compound_list is not None:
            py_logger.debug("Checking train, val and test values in init.")
            self.normalize_train_val_test()

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

    def normalize_train_val_test(self):
        """Normalize the train, val and test values."""
        py_logger.debug("Train, val and test are integers.")

        self.total_train = self.n_compounds - self.val - self.test - self.retrieval

        if self.train < 0:
            self.train = self.total_train

        if (total := self.total_train + self.val + self.test + self.retrieval) > self.n_compounds:
            py_logger.warning(f"Total split size ({total}) is larger than the dataset size ({self.n_compounds}).")
            raise ValueError(f"Total split size ({total}) is larger than the dataset size ({self.n_compounds}).")

        py_logger.debug(f"Train: {self.train}, val: {self.val}, test: {self.test}, retrieval: {self.retrieval}")

    @abstractmethod
    def split(self) -> Dict[str, List[str]]:
        """Split the data into total_train, val and test sets."""
        raise NotImplementedError

    @abstractmethod
    def split_train(self) -> List[str]:
        """Subsplit a train list into a smaller train list."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(train={self.train}, val={self.val}, test={self.test})"

    def __call__(
        self,
    ) -> Union[Tuple[List[str], List[str], List[str]], Tuple[List[str], List[str], List[str], List[str]]]:
        return self.split()
