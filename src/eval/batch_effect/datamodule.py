import os.path as osp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.modules.collate_fn import default_collate
from src.utils.io import load_image_paths_to_array


class BatchEffectDataset(Dataset):
    def __init__(
        self,
        target_load_df: pd.DataFrame,
        target_to_num: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        target_col: str = "target",
        channels: Optional[List[str]] = None,
        data_root_dir: Optional[str] = None,
    ):
        super().__init__()

        self.target_load_df = target_load_df
        self.target_col = target_col
        self.channels = channels or ["DNA", "AGP", "ER", "Mito", "RNA"]
        self.data_root_dir = data_root_dir

        if data_root_dir:
            for channel in self.channels:
                self.target_load_df.loc[:, f"FileName_Orig{channel}"] = self.target_load_df[
                    f"FileName_Orig{channel}"
                ].str.replace("/projects/", data_root_dir)

        self.targets = self.target_load_df[self.target_col].unique()
        self.targets.sort()
        self.target_to_num = target_to_num or {target: i for i, target in enumerate(self.targets)}
        self.n_targets = len(self.targets)

        self.transform = transform

        self.n_images = len(self.target_load_df)

    def __len__(self):
        return self.n_images

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_targets={self.n_targets}, n_images={self.n_images})"

    def __getitem__(self, idx: int):
        row = self.target_load_df.iloc[idx]

        labels = self.target_to_num[row[self.target_col]]

        img_paths = [row[f"FileName_Orig{channel}"] for channel in self.channels]
        img_array = load_image_paths_to_array(img_paths)  # A numpy array: (5, 768, 768)
        img_array = torch.from_numpy(img_array)
        if self.transform:
            img_array = self.transform(img_array)

        return {"label": labels, "image": img_array}


class BatchEffectDataModule(LightningDataModule):
    def __init__(
        self,
        target_load_df_path: str,
        split_path: str,
        split_type: str,  # random, plate_aware, source_aware
        label_col: str = "target",
        val_size: float = 0.1,
        test_size: float = 0.2,
        transform: Optional[Callable] = None,
        channels: Optional[List[str]] = None,
        data_root_dir: Optional[str] = None,
        collate_fn: Optional[Callable] = default_collate,
        batch_size: int = 256,
        num_workers: int = 16,
        pin_memory: bool = False,
        prefetch_factor: int = 3,
        drop_last: bool = False,
        metadata_path: str = "/projects/cpjump1/jump/metadata",
        load_data_path: str = "/projects/cpjump1/jump/load_data",
        random_state: int = 42,
    ):
        super().__init__()

        # paths
        self.target_load_df_path = target_load_df_path
        self.split_path = split_path
        self.split_type = split_type

        # for prepare_data
        self.metadata_path = metadata_path
        self.load_data_path = load_data_path
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        # dataset args
        self.label_col = label_col
        self.channels = channels
        self.data_root_dir = data_root_dir
        self.transform = transform

        # dataloader args
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

        # needed attributes
        self.target_load_df: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[BatchEffectDataset] = None
        self.val_dataset: Optional[BatchEffectDataset] = None
        self.test_dataset: Optional[BatchEffectDataset] = None

    @staticmethod
    def write_list_to_file(file_path: str, a_list: List[str]) -> None:
        with open(file_path, "w") as f:
            for item in a_list:
                f.write(f"{item}\n")

    @staticmethod
    def aware_split(df, col, test_size=0.2):
        unique_values = df[col].unique()
        train_values = np.random.choice(unique_values, size=int(test_size * len(unique_values)), replace=False)

        train_idx = df[df[col].isin(train_values)].index.tolist()
        train_idx = np.random.permutation(train_idx).tolist()
        test_ids = df[~df[col].isin(train_values)].index.tolist()
        test_ids = np.random.permutation(test_ids).tolist()

        return train_idx, test_ids

    def prepare_data(self) -> None:
        if not Path(self.target_load_df_path).exists():
            load_df = pd.read_parquet(osp.join(self.load_data_path, "final"))
            meta = pd.read_csv(osp.join(self.metadata_path, "local_metadata.csv"))
            target_local = meta.query("Metadata_PlateType == 'TARGET2'")
            target_meta = pd.read_csv(osp.join(self.metadata_path, "JUMP-Target-2_compound_metadata.tsv"), sep="\t")

            merged = pd.merge(
                target_meta.dropna(subset=["target"]),
                target_local,
                left_on=["InChIKey"],
                right_on=["Metadata_InChIKey"],
                how="inner",
            )[["target", "smiles", "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "pert_type"]]

            target_load_df = pd.merge(
                load_df,
                merged,
                on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
                how="inner",
            )

            Path(self.target_load_df_path).parent.mkdir(parents=True, exist_ok=True)

            target_load_df.to_csv(self.target_load_df_path, index=False)

        if not Path(self.split_path).exists():
            if "target_load_df" not in locals():
                target_load_df = pd.read_csv(self.target_load_df_path)

            # Create split dir
            Path(self.split_path).mkdir(parents=True, exist_ok=True)

            # Create Random Split from the target_load_df
            index = np.arange(len(target_load_df))
            random_dir = Path(self.split_path) / "random"
            random_dir.mkdir(parents=True, exist_ok=True)

            train_val_index, test_index = train_test_split(
                index, test_size=self.test_size, random_state=self.random_state, stratify=target_load_df[self.label_col]
            )
            train_index, val_index = train_test_split(
                train_val_index,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
                stratify=target_load_df[self.label_col].iloc[train_val_index],
            )

            self.write_list_to_file(osp.join(random_dir, "train.csv"), train_index)
            self.write_list_to_file(osp.join(random_dir, "val.csv"), val_index)
            self.write_list_to_file(osp.join(random_dir, "test.csv"), test_index)

            # Create Plate aware split from the target_load_df
            plate_dir = Path(self.split_path) / "plate_aware"
            plate_dir.mkdir(parents=True, exist_ok=True)

            train_val_index, test_index = self.aware_split(target_load_df, "Metadata_Plate", test_size=self.test_size)
            train_index, val_index = train_test_split(
                train_val_index,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
                stratify=target_load_df[self.label_col].iloc[train_val_index],
            )

            self.write_list_to_file(osp.join(plate_dir, "train.csv"), train_index)
            self.write_list_to_file(osp.join(plate_dir, "val.csv"), val_index)
            self.write_list_to_file(osp.join(plate_dir, "test.csv"), test_index)

            # Create Source aware split from the target_load_df
            source_dir = Path(self.split_path) / "source_aware"
            source_dir.mkdir(parents=True, exist_ok=True)

            train_val_index, test_index = self.aware_split(target_load_df, "Metadata_Source", test_size=self.test_size)
            train_index, val_index = train_test_split(
                train_val_index,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
                stratify=target_load_df[self.label_col].iloc[train_val_index],
            )

            self.write_list_to_file(osp.join(source_dir, "train.csv"), train_index)
            self.write_list_to_file(osp.join(source_dir, "val.csv"), val_index)
            self.write_list_to_file(osp.join(source_dir, "test.csv"), test_index)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.target_load_df is None:
            self.target_load_df = pd.read_csv(self.target_load_df_path)
            self.labels = self.target_load_df[self.label_col].unique().tolist()
            self.labels.sort()
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            self.num_to_labels = dict(enumerate(self.labels))

        if stage == "fit" or stage is None or stage == "validate":
            if self.train_dataset is None:
                train_df = pd.read_csv(
                    osp.join(self.split_path, self.split_type, "train.csv"), header=None, names=["index"]
                )
                self.train_dataset = BatchEffectDataset(
                    target_load_df=self.target_load_df.iloc[train_df["index"].tolist()],
                    target_to_num=self.label_to_idx,
                    label_col=self.label_col,
                    transform=self.transform,
                    channels=self.channels,
                    data_root_dir=self.data_root_dir,
                )
            if self.val_dataset is None:
                val_df = pd.read_csv(
                    osp.join(self.split_path, self.split_type, "val.csv"), header=None, names=["index"]
                )
                self.val_dataset = BatchEffectDataset(
                    target_load_df=self.target_load_df.iloc[val_df["index"].tolist()],
                    target_to_num=self.label_to_idx,
                    label_col=self.label_col,
                    transform=self.transform,
                    channels=self.channels,
                    data_root_dir=self.data_root_dir,
                )

        if stage == "test" and self.test_dataset is None:
            test_df = pd.read_csv(osp.join(self.split_path, self.split_type, "test.csv"), header=None, names=["index"])
            self.test_dataset = BatchEffectDataset(
                target_load_df=self.target_load_df.iloc[test_df["index"].tolist()],
                target_to_num=self.label_to_idx,
                label_col=self.label_col,
                transform=self.transform,
                channels=self.channels,
                data_root_dir=self.data_root_dir,
            )

    def phase_dataloader(self, phase: str) -> DataLoader:
        return DataLoader(
            dataset=getattr(self, f"{phase}_dataset"),
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            shuffle=(phase == "train"),
        )

    def train_dataloader(self) -> DataLoader:
        return self.phase_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.phase_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.phase_dataloader("test")

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
