"""Module containing the DataModules using the JUMP dataset."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence

import pandas as pd
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.models.jump_cl.dataset import MoleculeImageDataset
from src.modules.collate_fn import default_collate
from src.utils import color_log

py_logger = color_log.get_pylogger(__name__)


class SimpleRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        split_path: str,
        num_workers: int = 2,
        prefetch_factor: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = default_collate,
        compound_str_type: Optional[Literal["smiles", "inchi", "selfies"]] = None,
        compound_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        channels: Sequence[str] = ("DNA", "AGP", "ER", "Mito", "RNA"),
        check_compound_transform: bool = False,
        max_tries: int = 10,
        data_root_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["transform", "compound_transform", "collate_fn"])

        # load paths
        self.split_path = split_path
        self.retrieval_compound_path = Path(self.split_path) / "retrieval_compound_dict.json"
        self.retrieval_load_df_path = Path(self.split_path) / "retrieval_load_df.parquet"

        # data transformations
        self.transform = transform
        self.compound_transform = compound_transform
        self.compound_str_type = compound_str_type

        if self.compound_str_type and hasattr(self.compound_transform, "compound_str_type"):
            self.compound_transform.compound_str_type = self.compound_str_type

        # data loaders
        if collate_fn:
            self.collate_fn = collate_fn
        else:
            self.collate_fn = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # dataset
        self.retrieval_dataset = None
        # self.retrieval_ids_path = osp.join(split_path, "retrieval.csv")

        # dataset args
        self.data_root_dir = data_root_dir
        self.channels = channels
        self.col_fstring = "FileName_Orig{}"
        self.max_tries = max_tries
        self.image_sampler = sampler
        self.check_compound_transform = check_compound_transform

    def replace_root_dir(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.data_root_dir is not None:
            for channel in self.channels:
                df.loc[:, f"FileName_Orig{channel}"] = df[f"FileName_Orig{channel}"].str.replace(
                    "/projects/", self.data_root_dir
                )
        return df

    def setup(self, stage: Optional[str] = None) -> None:
        if (stage == "retrieval" or stage is None or stage == "predict") and self.retrieval_dataset is None:
            py_logger.info("Preparing retrieval dataset")
            print("Preparing retrieval dataset")

            retrieval_load_df = pd.read_parquet(Path(self.split_path) / "retrieval_load_df.parquet")
            retrieval_load_df = self.replace_root_dir(retrieval_load_df)

            with open(Path(self.split_path) / "retrieval_compound_dict.json") as handle:
                retrieval_compound_dict = json.load(handle)

            self.retrieval_dataset = MoleculeImageDataset(
                load_df=retrieval_load_df,
                compound_dict=retrieval_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.image_sampler,
                compound_str_type=self.compound_str_type,
                max_tries=self.max_tries,
            )

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self, **kwargs) -> DataLoader:
        retrieval_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "collate_fn": self.collate_fn,
        }
        retrieval_kwargs.update(kwargs)
        return DataLoader(
            dataset=self.retrieval_dataset,
            **retrieval_kwargs,
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
