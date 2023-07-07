# It is important to use scaffold splitting see https://www.oloren.ai/blog/scaff_split.html
# Take inspiration from https://github.com/snap-stanford/ogb/

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split

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

        if self.compound_list is not None:
            py_logger.debug("Checking train, val and test values in init.")
            self.normalize_train_val_test()

    def set_compound_list(self, compound_list: List[str]):
        """Set the compound list."""
        self.compound_list = compound_list
        py_logger.debug("Checking train, val and test values in set_compound_list.")
        self.normalize_train_val_test()

    def normalize_train_val_test(self):
        """Normalize the train, val and test values."""
        total = self.train + self.val + self.test
        if isinstance(self.train, int) and isinstance(self.val, int) and isinstance(self.test, int):
            py_logger.debug("Train, val and test are integers.")
            if self.compound_list is not None and total > len(self.compound_list):
                py_logger.warning(
                    f"Total split size ({total}) is larger than the dataset size ({len(self.compound_list)})."
                )
                py_logger.warning("Normalizing train, val and test to 1.")
                self.train = self.train / total
                self.val = self.val / total
                self.test = self.test / total
        if isinstance(self.train, float) and isinstance(self.val, float) and isinstance(self.test, float):
            py_logger.debug("Train, val and test are floats. Normalizing to 1.")
            self.train = self.train / total
            self.val = self.val / total
            self.test = self.test / total
        else:
            raise ValueError("Train, val and test must be either integers or floats.")

        py_logger.debug(f"Train: {self.train:.2g}, val: {self.val:.2g}, test: {self.test:.2g}")

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

    def __init__(
        self,
        train: Union[float, int],
        val: Union[float, int],
        test: Union[float, int],
        compound_list: Optional[List[str]] = None,
    ):
        super().__init__(train, val, test, compound_list)

    def split(self) -> Tuple[List[str], List[str], List[str]]:
        """Split the data into train, val and test sets."""
        train_val, test = train_test_split(self.compound_list, test_size=self.test, random_state=42)
        train, val = train_test_split(train_val, test_size=self.val / (self.train + self.val), random_state=42)

        return train, val, test
