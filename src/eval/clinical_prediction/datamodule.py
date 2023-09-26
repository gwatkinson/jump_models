"""Custom dataset for the OGB evaluation tasks."""

# flake8: noqa

from typing import Any, Callable, Dict, List, Literal, Optional

import datamol
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.modules.collate_fn import SmilesList, default_collate
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class HintClinicalDataset(Dataset):
    def __init__(
        self,
        phase_df: pd.DataFrame,
        smiless_col: str = "smiless",
        label_col: str = "label",
    ):
        self.df = phase_df
        self.smiless_col = smiless_col
        self.label_col = label_col

        self.valid_df = self.validate_df()

        self.smiles = self.valid_df["valid_smiles"]
        self.targets = self.valid_df[self.label_col]

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def smiles_txt_to_lst(text):
        text = text[1:-1]
        lst = [i.strip()[1:-1] for i in text.split(",")]
        return lst

    def check_smiles_list(self, smiles_list_str):
        smiles_list = self.smiles_txt_to_lst(smiles_list_str)
        valid_smiles = []
        for smiles in smiles_list:
            mol = datamol.to_mol(smiles)
            if smiles in ["[Cl-].[Na+]", "O", "[Se]", "[Li+]"]:
                continue
            if mol is not None:
                valid_smiles.append(datamol.to_smiles(mol))

        return valid_smiles

    def validate_df(self):
        smiles_to_test = self.df[self.smiless_col].values.tolist()

        valid_smiles = []

        for smiles_list in smiles_to_test:
            valid_smiles.append(self.check_smiles_list(smiles_list))

        self.df["valid_smiles"] = valid_smiles

        return self.df[self.df["valid_smiles"].apply(len) != 0]

    def __getitem__(self, index):
        label = self.targets.iloc[index]
        smiles = SmilesList(self.smiles.iloc[index])

        return {
            "smiles_list": smiles,
            "label": label,
        }


class HintClinicalDataModule(LightningDataModule):
    default_phase = Optional[Literal["I", "II", "III"]] = None

    def __init__(
        self,
        hint_dir: str,
        phase: Optional[Literal["I", "II", "III"]] = None,
        collate_fn: Optional[Callable] = default_collate,
        smiless_col: str = "smiless",
        label_col: str = "label",
        batch_size: int = 256,
        num_workers: int = 16,
        pin_memory: bool = False,
        prefetch_factor: int = 3,
        drop_last: bool = False,
    ):
        super().__init__()

        self.hint_dir = hint_dir
        self.phase = phase or self.default_phase

        # dataset args
        self.smiless_col = smiless_col
        self.label_col = label_col

        # dataloader args
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

        # attributes
        self.train_dataset: Optional[HintClinicalDataset] = None
        self.val_dataset: Optional[HintClinicalDataset] = None
        self.test_dataset: Optional[HintClinicalDataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None or stage == "validate":
            train_df = pd.read_csv(f"{self.hint_dir}/phase_{self.phase}_train.csv")
            self.train_dataset = HintClinicalDataset(
                train_df,
                smiless_col=self.smiless_col,
                label_col=self.label_col,
            )

            val_df = pd.read_csv(f"{self.hint_dir}/phase_{self.phase}_valid.csv")
            self.val_dataset = HintClinicalDataset(
                val_df,
                smiless_col=self.smiless_col,
                label_col=self.label_col,
            )

        if stage == "test" or stage is None:
            test_df = pd.read_csv(f"{self.hint_dir}/phase_{self.phase}_test.csv")
            self.test_dataset = HintClinicalDataset(
                test_df,
                smiless_col=self.smiless_col,
                label_col=self.label_col,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class HintClinicalDataModulePhaseI(HintClinicalDataModule):
    default_phase = "I"


class HintClinicalDataModulePhaseII(HintClinicalDataModule):
    default_phase = "II"


class HintClinicalDataModulePhaseIII(HintClinicalDataModule):
    default_phase = "III"
