from typing import Dict, List

from sklearn.model_selection import train_test_split

from src.splitters import BaseSplitter
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class RandomSplitter(BaseSplitter):
    """Split the data into train, val and test sets using the scaffold
    splitting method."""

    def split(self) -> Dict[str, List[str]]:
        """Split the data into train, val and test sets."""
        if self.compound_list is None:
            raise ValueError("Compound list is None.")

        py_logger.info("Generating sets ...")

        train_val_retrieval, test_cpds = train_test_split(
            self.compound_list,
            test_size=self.test,
            random_state=self.random_state,
        )
        train_val, retrieval_cpds = train_test_split(
            train_val_retrieval, test_size=self.retrieval, random_state=self.random_state
        )
        total_train_cpds, val_cpds = train_test_split(train_val, test_size=self.val, random_state=self.random_state)

        output = {
            "total_train": total_train_cpds,
            "val": val_cpds,
            "test": test_cpds,
            "retrieval": retrieval_cpds,
        }

        self.total_train_cpds = total_train_cpds
        self.val_cpds = val_cpds
        self.test_cpds = test_cpds
        self.retrieval_cpds = retrieval_cpds

        return output

    def split_train(self, total_train_cpds=None) -> List[str]:
        total_train_cpds = total_train_cpds or self.total_train_cpds

        train_cpds, _ = train_test_split(
            total_train_cpds,
            train_size=self.train,
            random_state=self.random_state,
        )
        self.train_cpds = train_cpds
        return train_cpds
