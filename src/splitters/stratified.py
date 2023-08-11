from typing import Dict, List

from sklearn.model_selection import train_test_split

from src.splitters import BaseSplitter


class StratifiedSplitter(BaseSplitter):
    def split(self) -> Dict[str, List[str]]:
        """Split the data into train, val and test sets."""
        if self.compound_list is None:
            raise ValueError("Label list is None.")

        ids = list(range(len(self.compound_list)))

        if self.input_type == "int":
            train_val, test = train_test_split(
                ids,
                test_size=self.test,
                train_size=self.train + self.val,
                random_state=self.random_state,
                stratify=self.compound_list,
            )
            sub_labels = [self.compound_list[i] for i in train_val]
            train, val = train_test_split(
                train_val,
                test_size=self.val,
                train_size=self.train,
                random_state=self.random_state,
                stratify=sub_labels,
            )
        elif self.normalized:
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
        else:
            raise ValueError("Error in the code.")

        return {"train": train, "val": val, "test": test}
