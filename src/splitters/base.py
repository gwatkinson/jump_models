from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class BaseSplitter(ABC):
    """Base class for splitters.

    It requires a train, val, split decomposition.
    It can be in the float, float, float format in which case the values are normalized to 1 and correspond to the
    fractions of the entire dataset.
    Else, it can be in the int, int, int format in which case the values are the number of compounds in each split.
    If the sum is larger than the number of compounds in the dataset, a warning is raised and the values are normalized.
    You can also pass a list of compounds to split during the initialization or later using the set_compound_list
    function.

    Args:
        train (Union[float, int]): Train split.
        val (Union[float, int]): Validation split.
        test (Union[float, int]): Test split.
        compound_list (Optional[List[str]]): List of compounds to split.

    Properties:
        input_type (str): Either "int" or "float" depending on the input type of the train, val and test values.
        n_compounds (int): Number of compounds in the compound list.
        train (Union[float, int]): Train split value.
        test (Union[float, int]): Test split value.
        val (Union[float, int]): Validation split value.
        normalized (bool): Whether the train, val and test values have been normalized to 1.

    Methods:
        set_compound_list: Set the compound list.
        normalize_train_val_test: Normalize the train, val and test values depending on the input type.
        split: Split the data into train, val and test sets.

    Raises:
        ValueError: If the train, val and test values are not either integers or floats.
        ValueError: If the compound list is None.

    Examples:
        >>> from jump_models.src.data.splits import BaseSplitter
        >>> splitter = BaseSplitter(train=0.8, val=0.1, test=0.1)
        >>> compound_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        >>> splitter.set_compound_list(compound_list)
        >>> splitter.split()
        (['c', 'b', 'd', 'a', 'f', 'h', 'j', 'e'], ['i'], ['g'])
        >>> splitter.normalized
        True
        >>> splitter.train, splitter.val, splitter.test
        0.8, 0.1, 0.1

        >>> splitter = BaseSplitter(train=4, val=1, test=1)
        >>> splitter.set_compound_list(compound_list)
        >>> splitter.split()
        (['c', 'b', 'd', 'a'], ['f'], ['h'])
        >>> splitter.normalized
        False
        >>> splitter.train, splitter.val, splitter.test
        4, 1, 1

        >>> splitter = BaseSplitter(train=16, val=2, test=2)
        >>> splitter.set_compound_list(compound_list)
        >>> splitter.split()
        (['c', 'b', 'd', 'a', 'f', 'h', 'j', 'e'], ['i'], ['g'])
        >>> splitter.normalized
        True
        >>> splitter.train, splitter.val, splitter.test
        0.8, 0.1, 0.1
    """

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
        return f"{self.__class__.__name__}(train={self.train}, val={self.val}, test={self.test}, len_compound_list={self.n_compounds})"

    def __call__(
        self,
    ) -> Union[Tuple[List[str], List[str], List[str]], Tuple[List[str], List[str], List[str], List[str]]]:
        return self.split()
