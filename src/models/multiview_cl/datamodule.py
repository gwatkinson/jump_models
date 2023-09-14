"""Module containing the DataModules using the JUMP dataset."""

import json
import os.path as osp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.models.jump_cl.dataset import MoleculeImageDataset
from src.models.multiview_cl.dataset import MultiviewDataset
from src.modules.collate_fn import default_collate
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class MultiviewDataModule(LightningDataModule):
    dataset_cls = MultiviewDataset  # The dataset class to use

    def __init__(
        self,
        split_path: str,
        dataloader_config: DictConfig,
        n_views: int = 3,
        train_ids_name: str = "train",
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = default_collate,
        data_root_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["transform", "compound_transform", "collate_fn"])

        # metadata
        self.load_df: Optional[pd.DataFrame] = None
        self.compound_dict: Optional[Dict[str, List[str]]] = None
        self.data_root_dir = data_root_dir

        self.n_views = n_views

        # data transformations
        self.transform = transform
        self.compound_transform = compound_transform
        if collate_fn:
            self.collate_fn = collate_fn
        else:
            self.collate_fn = None

        # datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.retrieval_dataset: Optional[Dataset] = None

        # data loaders
        self.dataloader_config = dataloader_config

        # split paths
        self.split_path = split_path
        self.total_train_ids_path = osp.join(split_path, "total_train.csv")
        self.train_ids_name = train_ids_name
        self.train_ids_path = osp.join(split_path, f"{train_ids_name}.csv")
        self.val_ids_path = osp.join(split_path, "val.csv")
        self.test_ids_path = osp.join(split_path, "test.csv")
        self.retrieval_ids_path = osp.join(split_path, "retrieval.csv")

        # kwargs
        self.channels = kwargs.get("channels", ["DNA", "AGP", "ER", "Mito", "RNA"])
        self.col_fstring = kwargs.get("col_fstring", "FileName_Orig{channel}")

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def replace_root_dir(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.data_root_dir is not None:
            for channel in self.channels:
                df.loc[:, f"FileName_Orig{channel}"] = df[f"FileName_Orig{channel}"].str.replace(
                    "/projects/", self.data_root_dir
                )
        return df

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is None and (stage == "fit" or stage is None):
            py_logger.info("Preparing train dataset")

            train_load_df = pd.read_parquet(Path(self.split_path) / f"{self.train_ids_name}_load_df.parquet")
            train_load_df = self.replace_root_dir(train_load_df)

            with open(Path(self.split_path) / f"{self.train_ids_name}_compound_dict.json") as handle:
                train_compound_dict = json.load(handle)

            self.train_dataset = self.dataset_cls(
                load_df=train_load_df,
                compound_dict=train_compound_dict,
                n_views=self.n_views,
                transform=self.transform,
                compound_transform=self.compound_transform,
                channels=self.channels,
                col_fstring=self.col_fstring,
            )

        if self.val_dataset is None and (stage == "fit" or stage == "evaluate" or stage is None):
            py_logger.info("Preparing validation dataset")

            val_load_df = pd.read_parquet(Path(self.split_path) / "val_load_df.parquet")
            val_load_df = self.replace_root_dir(val_load_df)

            with open(Path(self.split_path) / "val_compound_dict.json") as handle:
                val_compound_dict = json.load(handle)

            self.val_dataset = self.dataset_cls(
                load_df=val_load_df,
                compound_dict=val_compound_dict,
                n_views=self.n_views,
                transform=self.transform,
                compound_transform=self.compound_transform,
                channels=self.channels,
                col_fstring=self.col_fstring,
            )

        if stage == "test" and self.test_dataset is None:
            py_logger.info("Preparing test dataset")

            test_load_df = pd.read_parquet(Path(self.split_path) / "test_load_df.parquet")
            test_load_df = self.replace_root_dir(test_load_df)

            with open(Path(self.split_path) / "test_compound_dict.json") as handle:
                test_compound_dict = json.load(handle)

            self.test_dataset = MoleculeImageDataset(
                load_df=test_load_df,
                compound_dict=test_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                channels=self.channels,
                col_fstring=self.col_fstring,
            )

        if stage == "retrieval" and self.retrieval_dataset is None:
            py_logger.info("Preparing retrieval dataset")

            retrieval_load_df = pd.read_parquet(Path(self.split_path) / "retrieval_load_df.parquet")
            retrieval_load_df = self.replace_root_dir(retrieval_load_df)

            with open(Path(self.split_path) / "retrieval_compound_dict.json") as handle:
                retrieval_compound_dict = json.load(handle)

            self.retrieval_dataset = MoleculeImageDataset(
                load_df=retrieval_load_df,
                compound_dict=retrieval_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                channels=self.channels,
                col_fstring=self.col_fstring,
            )

    def train_dataloader(self, **kwargs) -> DataLoader:
        train_kwargs = OmegaConf.to_container(self.dataloader_config.train, resolve=True)
        train_kwargs.update(kwargs)
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            **train_kwargs,
        )

    def val_dataloader(self, **kwargs) -> DataLoader:
        val_kwargs = OmegaConf.to_container(self.dataloader_config.val, resolve=True)
        val_kwargs.update(kwargs)
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collate_fn,
            **val_kwargs,
        )

    def test_dataloader(self, **kwargs) -> DataLoader:
        test_kwargs = OmegaConf.to_container(self.dataloader_config.test, resolve=True)
        test_kwargs["batch_size"] = 100  # For 1/100 retrieval
        test_kwargs.update(kwargs)
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collate_fn,
            **test_kwargs,
        )

    def retrieval_dataloader(self, **kwargs) -> DataLoader:
        retrieval_kwargs = OmegaConf.to_container(self.dataloader_config.test, resolve=True)
        retrieval_kwargs["batch_size"] = 100  # For 1/100 retrieval
        retrieval_kwargs.update(kwargs)
        return DataLoader(
            dataset=self.retrieval_dataset,
            collate_fn=self.collate_fn,
            **retrieval_kwargs,
        )

    def predict_dataloader(self, **kwargs) -> DataLoader:
        return self.retrieval_dataloader(**kwargs)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
