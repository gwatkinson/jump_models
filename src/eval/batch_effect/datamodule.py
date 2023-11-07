import os.path as osp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.modules.collate_fn import default_collate
from src.utils.io import load_image_paths_to_array


class BatchEffectDataset(Dataset):
    def __init__(
        self,
        target_load_df: pd.DataFrame,
        target_to_num: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        channels: Optional[List[str]] = None,
        data_root_dir: Optional[str] = None,
        label_col: str = "target",
        source_col: str = "Metadata_Source",
        batch_col: str = "Metadata_Batch",
        plate_col: str = "Metadata_Plate",
        well_col: str = "Metadata_Well",
    ):
        super().__init__()

        self.target_load_df = target_load_df
        self.label_col = label_col
        self.source_col = source_col
        self.batch_col = batch_col
        self.plate_col = plate_col
        self.well_col = well_col

        self.channels = channels or ["DNA", "AGP", "ER", "Mito", "RNA"]
        self.data_root_dir = data_root_dir

        if data_root_dir:
            for channel in self.channels:
                self.target_load_df.loc[:, f"FileName_Orig{channel}"] = self.target_load_df[
                    f"FileName_Orig{channel}"
                ].str.replace("/projects/", data_root_dir)

        self.targets = self.target_load_df[self.label_col].unique()
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

        # labels = self.target_to_num[row[self.label_col]]
        labels = row[self.label_col]
        sources = row[self.source_col]
        batches = row[self.batch_col]
        plates = row[self.plate_col]
        wells = row[self.well_col]

        img_paths = [row[f"FileName_Orig{channel}"] for channel in self.channels]
        img_array = load_image_paths_to_array(img_paths)  # A numpy array: (5, 768, 768)
        img_array = torch.from_numpy(img_array)
        if self.transform:
            img_array = self.transform(img_array)

        return {
            "label": labels,
            "image": img_array,
            "source": sources,
            "batch": batches,
            "plate": plates,
            "well": wells,
        }


class BatchEffectDMSODataset(Dataset):
    def __init__(
        self,
        target_load_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        channels: Optional[List[str]] = None,
        data_root_dir: Optional[str] = None,
        source_col: str = "Metadata_Source",
        batch_col: str = "Metadata_Batch",
        plate_col: str = "Metadata_Plate",
        well_col: str = "Metadata_Well",
    ):
        super().__init__()

        self.target_load_df = target_load_df
        self.source_col = source_col
        self.batch_col = batch_col
        self.plate_col = plate_col
        self.well_col = well_col

        self.channels = channels or ["DNA", "AGP", "ER", "Mito", "RNA"]
        self.data_root_dir = data_root_dir

        if data_root_dir:
            for channel in self.channels:
                self.target_load_df.loc[:, f"FileName_Orig{channel}"] = self.target_load_df[
                    f"FileName_Orig{channel}"
                ].str.replace("/projects/", data_root_dir)

        self.transform = transform

        self.n_images = len(self.target_load_df)

    def __len__(self):
        return self.n_images

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_images={self.n_images})"

    def __getitem__(self, idx: int):
        row = self.target_load_df.iloc[idx]

        # labels = self.target_to_num[row[self.label_col]]
        sources = row[self.source_col]
        batches = row[self.batch_col]
        plates = row[self.plate_col]
        wells = row[self.well_col]

        img_paths = [row[f"FileName_Orig{channel}"] for channel in self.channels]
        img_array = load_image_paths_to_array(img_paths)  # A numpy array: (5, 768, 768)
        img_array = torch.from_numpy(img_array)
        if self.transform:
            img_array = self.transform(img_array)

        return {
            "label": "DMSO",
            "image": img_array,
            "source": sources,
            "batch": batches,
            "plate": plates,
            "well": wells,
        }


class TotalBatchEffectDataModule(LightningDataModule):
    def __init__(
        self,
        target_load_df_path: str,
        dmso_load_df_path: str,
        subset_targets: Optional[int] = None,
        label_col: str = "target",
        source_col: str = "Metadata_Source",
        batch_col: str = "Metadata_Batch",
        plate_col: str = "Metadata_Plate",
        well_col: str = "Metadata_Well",
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
        self.dmso_load_df_path = dmso_load_df_path

        # for prepare_data
        self.metadata_path = metadata_path
        self.load_data_path = load_data_path
        self.random_state = random_state
        self.subset_targets = subset_targets

        # dataset args
        self.label_col = label_col
        self.source_col = source_col
        self.batch_col = batch_col
        self.plate_col = plate_col
        self.well_col = well_col
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
        self.predict_dataset: Optional[BatchEffectDataset] = None
        self.dmso_dataset: Optional[BatchEffectDMSODataset] = None

    def prepare_data(self) -> None:
        if not Path(self.target_load_df_path).exists() or not Path(self.dmso_load_df_path).exists():
            print("Preparing datamodule...")
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

            dmso_meta = meta.query("trt=='target_neg' & Metadata_PlateType == 'TARGET2'")[
                ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
            ]
            dmso_merge = pd.merge(
                dmso_meta,
                load_df,
                on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
                how="inner",
            )

            Path(self.dmso_load_df_path).parent.mkdir(parents=True, exist_ok=True)
            dmso_merge.to_csv(self.dmso_load_df_path, index=False)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.target_load_df is None:
            self.target_load_df = pd.read_csv(self.target_load_df_path)

            if self.subset_targets:
                targets = self.target_load_df.target.value_counts()
                sub_targets = targets.index[:10].tolist()

                self.target_load_df = self.target_load_df[self.target_load_df["target"].isin(sub_targets)]

            self.labels = self.target_load_df[self.label_col].unique().tolist()
            self.labels.sort()
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            self.num_to_labels = dict(enumerate(self.labels))

        if self.predict_dataset is None:
            self.predict_dataset = BatchEffectDataset(
                target_load_df=self.target_load_df,
                target_to_num=self.label_to_idx,
                transform=self.transform,
                channels=self.channels,
                data_root_dir=self.data_root_dir,
                label_col=self.label_col,
                source_col=self.source_col,
                batch_col=self.batch_col,
                plate_col=self.plate_col,
                well_col=self.well_col,
            )

        if self.dmso_dataset is None:
            dmso_df = pd.read_csv(self.dmso_load_df_path)
            self.dmso_dataset = BatchEffectDMSODataset(
                target_load_df=dmso_df,
                transform=self.transform,
                channels=self.channels,
                data_root_dir=self.data_root_dir,
                source_col=self.source_col,
                batch_col=self.batch_col,
                plate_col=self.plate_col,
                well_col=self.well_col,
            )

    def predict_dataloader(self, **kwargs) -> DataLoader:
        args = {
            "collate_fn": self.collate_fn,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "drop_last": self.drop_last,
            "shuffle": False,
        }
        args.update(kwargs)
        return DataLoader(dataset=self.predict_dataset, **args)

    def dmso_dataloader(self, **kwargs):
        args = {
            "collate_fn": self.collate_fn,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "drop_last": self.drop_last,
            "shuffle": False,
        }
        args.update(kwargs)
        return DataLoader(dataset=self.dmso_dataset, **args)

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
