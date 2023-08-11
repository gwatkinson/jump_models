from typing import Dict, List

from sklearn.model_selection import train_test_split

from src.splitters import BaseSplitter


class RandomSplitter(BaseSplitter):
    """Split data randomly into train, val and test sets."""

    def split(self) -> Dict[str, List[str]]:
        """Split the data into train, val and test sets."""
        if self.compound_list is None:
            raise ValueError("Compound list is None.")

        if self.input_type == "int":
            train_val, test = train_test_split(
                self.compound_list,
                test_size=self.test,
                train_size=self.train + self.val,
                random_state=self.random_state,
            )
            train, val = train_test_split(
                train_val, test_size=self.val, train_size=self.train, random_state=self.random_state
            )
        elif self.normalized:
            train_val, test = train_test_split(self.compound_list, test_size=self.test, random_state=self.random_state)
            train, val = train_test_split(
                train_val, test_size=self.val / (self.train + self.val), random_state=self.random_state
            )
        else:
            raise ValueError("Error in the code.")

        return {"train": train, "val": val, "test": test}
