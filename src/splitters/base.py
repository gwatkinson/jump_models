import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

py_logger = logging.getLogger(__name__)


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
