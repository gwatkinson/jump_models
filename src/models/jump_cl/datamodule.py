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
from src.utils import pylogger
from src.utils.io import load_load_df_from_parquet, load_metadata_df_from_csv

py_logger = pylogger.get_pylogger(__name__)


class BasicJUMPDataModule(LightningDataModule):
    """Basic LightningDataModule for the JUMP dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    dataset_cls = MoleculeImageDataset  # The dataset class to use

    def __init__(
        self,
        image_metadata_path: str,
        compound_metadata_path: str,
        split_path: str,
        dataloader_config: DictConfig,
        force_split: bool = False,
        compound_col: str = "Metadata_InChI",
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        image_sampler: Optional[Callable[[List[str]], str]] = None,
        use_compond_cache: bool = False,
        data_root_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize a BasicJUMPDataModule instance.

        Create a LightningDataModule that will load the JUMP dataset from the provided paths.
        This class provides the dataset and dataloaders for the JUMP dataset.

        Args:
            image_metadata_path (str): Path to the image metadata parquet file.
                This df contains the path to the images, and the index should be the same as the index_str.
            compound_metadata_path (str): Path to the compound dict json file.
                This dict contains the compound names as keys and the list of indexes with the compound as values.
            split_path (str): Path to the directory containing the train, test and val split csvs.
            dataloader_config (DictConfig): Config dict for the dataloaders.
                Should contains the train, test, val keys with subkeys giving the dataloader parameters.
            force_split (bool): Whether to force the split of the data.
                If False, will try to load the split csvs from the split_path.
            compound_col (str): Name of the column containing the compound names in the image metadata df.
            transform (Optional[Callable]): Transform to apply to the images.
            compound_transform (Optional[Callable]): Transform to apply to the compounds.
            image_sampler (Optional[Callable[[List[str]], str]]): Function to use to sample images from a list of images.
            **kwargs: Additional arguments used during the prepare_data method or for the dataset_cls.
                Those are:
                    index_str (str): Format string to use to create the index of the image metadata df.
                        The format string should contain the following keys:
                            Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well, Metadata_Site
                    metadata_dir (str): Path to the metadata csv file.
                    local_load_data_dir (str): Path to the local load data directory.
                    splitter (Optional[BaseSplitter]): Function to use to split the data.
                    channels (List[str]): List of channels to use.
                    col_fstring (str): Format string to use to create the column names of the image metadata df from the channel.
                    id_cols (List[str]): List of columns to use as ids.
                    extra_cols (Optional[List[str]]): List of extra columns to use.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["transform", "compound_transform", "collate_fn"])

        # metadata
        self.load_df: Optional[pd.DataFrame] = None
        self.compound_dict: Optional[Dict[str, List[str]]] = None
        self.data_root_dir = data_root_dir

        # split ids
        self.train_cpds: Optional[List] = None
        self.test_cpds: Optional[List] = None
        self.val_cpds: Optional[List] = None

        # data transformations
        self.transform = transform
        self.compound_transform = compound_transform
        self.collate_fn = collate_fn

        # datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # data loaders
        self.dataloader_config = dataloader_config
        self.image_sampler = image_sampler

        # other
        self.compound_col = compound_col
        self.split_path = split_path
        self.image_metadata_path = image_metadata_path
        self.compound_metadata_path = compound_metadata_path
        self.use_compond_cache = use_compond_cache

        # split paths
        self.force_split = force_split
        self.train_ids_path = osp.join(split_path, "train_ids.csv")
        self.val_ids_path = osp.join(split_path, "val_ids.csv")
        self.test_ids_path = osp.join(split_path, "test_ids.csv")
        self.retrieval_ids_path = osp.join(split_path, "retrieval_ids.csv")

        # kwargs
        self.index_str = kwargs.get(
            "index_str", "{Metadata_Source}__{Metadata_Batch}__{Metadata_Plate}__{Metadata_Well}__{Metadata_Site}"
        )
        self.metadata_dir = kwargs.get("metadata_dir", "../cpjump1/jump/metadata/complete_metadata.csv")
        self.local_load_data_dir = kwargs.get("local_load_data_dir", "../cpjump1/jump/load_data/final/")
        self.splitter = kwargs.get("splitter")
        self.channels = kwargs.get("channels", ["DNA", "AGP", "ER", "Mito", "RNA"])
        self.col_fstring = kwargs.get("col_fstring", "FileName_Orig{channel}")
        self.id_cols = kwargs.get("id_cols", ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
        self.extra_cols = kwargs.get("extra_cols", ["Metadata_PlateType", "Metadata_Site"])

    def prepare_load_df_with_meta(self):
        img_path = Path(self.image_metadata_path)
        load_dir = Path(self.local_load_data_dir)
        meta_dir = Path(self.metadata_dir)
        f_string = self.index_str
        cols_to_keep = (
            self.id_cols
            + [self.col_fstring.format(channel=channel) for channel in self.channels]
            + [self.compound_col]
            + self.extra_cols
        )

        py_logger.info("Preparing image metadata")
        py_logger.debug(f"{img_path} does not exist.")
        py_logger.debug(f"Loading load data df from {load_dir} ...")
        load_df = load_load_df_from_parquet(load_dir)

        py_logger.debug(f"Loading metadata df from {meta_dir} ...")
        meta_df = load_metadata_df_from_csv(meta_dir)

        py_logger.debug(f"ID cols: {self.id_cols}")
        py_logger.debug(f"Extra cols: {self.extra_cols}")
        py_logger.debug(f"load_df cols: {load_df.columns.tolist()}")
        py_logger.debug(f"meta_df cols: {meta_df.columns.tolist()}")

        py_logger.info("Merging metadata and load data...")
        load_df_with_meta = load_df.merge(meta_df, how="left", on=self.id_cols).dropna(subset=[self.compound_col])
        load_df_with_meta = load_df_with_meta.query("Metadata_PlateType == 'COMPOUND'")
        load_df_with_meta["index"] = load_df_with_meta.apply(lambda x: f_string.format(**x), axis=1)

        load_df_with_meta = load_df_with_meta.set_index("index", drop=True).loc[:, cols_to_keep]

        py_logger.debug(f"load_df_with_meta ids unique: {load_df_with_meta.index.is_unique}")

        py_logger.debug(f"Saving image metadata df to {img_path} ...")
        img_path.parent.mkdir(exist_ok=True, parents=True)
        load_df_with_meta.to_parquet(
            path=str(img_path),
            engine="pyarrow",
            compression="snappy",
            index=True,
        )

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        img_path = Path(self.image_metadata_path)
        comp_path = Path(self.compound_metadata_path)
        train_ids_path = Path(self.train_ids_path)
        test_ids_path = Path(self.test_ids_path)
        val_ids_path = Path(self.val_ids_path)
        retrieval_ids_path = Path(self.retrieval_ids_path)
        load_dir = Path(self.local_load_data_dir)
        meta_dir = Path(self.metadata_dir)
        f_string = self.index_str
        cols_to_keep = (
            self.id_cols
            + [self.col_fstring.format(channel=channel) for channel in self.channels]
            + [self.compound_col]
            + self.extra_cols
        )

        # Prepare image metadata
        if not img_path.exists():
            py_logger.info("Preparing image metadata")
            py_logger.debug(f"{img_path} does not exist.")
            py_logger.debug(f"Loading load data df from {load_dir} ...")
            load_df = load_load_df_from_parquet(load_dir)

            py_logger.debug(f"Loading metadata df from {meta_dir} ...")
            meta_df = load_metadata_df_from_csv(meta_dir)

            py_logger.debug(f"ID cols: {self.id_cols}")
            py_logger.debug(f"Extra cols: {self.extra_cols}")
            py_logger.debug(f"load_df cols: {load_df.columns.tolist()}")
            py_logger.debug(f"meta_df cols: {meta_df.columns.tolist()}")

            py_logger.info("Merging metadata and load data...")
            load_df_with_meta = load_df.merge(meta_df, how="left", on=self.id_cols).dropna(subset=[self.compound_col])
            load_df_with_meta = load_df_with_meta.query("Metadata_PlateType == 'COMPOUND'")
            load_df_with_meta["index"] = load_df_with_meta.apply(lambda x: f_string.format(**x), axis=1)

            load_df_with_meta = load_df_with_meta.set_index("index", drop=True).loc[:, cols_to_keep]

            py_logger.debug(f"load_df_with_meta ids unique: {load_df_with_meta.index.is_unique}")

            py_logger.debug(f"Saving image metadata df to {img_path} ...")
            img_path.parent.mkdir(exist_ok=True, parents=True)
            load_df_with_meta.to_parquet(
                path=str(img_path),
                engine="pyarrow",
                compression="snappy",
                index=True,
            )

        # Prepare compound metadata
        if not comp_path.exists():
            py_logger.info("Preparing compound metadata")
            py_logger.debug(f"{comp_path} does not exist.")

            if "load_df_with_meta" not in locals():
                py_logger.debug(f"Loading local load data df from {img_path} ...")
                load_df_with_meta = load_load_df_from_parquet(img_path)

            py_logger.info("Creating the compound dictionary...")
            compound_df = load_df_with_meta.groupby(self.compound_col).apply(lambda x: x.index.tolist())
            compound_dict = compound_df.to_dict()

            py_logger.debug(f"Saving compound dictionary to {comp_path} ...")
            with open(comp_path, "w") as handle:
                json.dump(compound_dict, handle)

        # Prepare train, test and val ids
        split_not_exists = not train_ids_path.exists() or not test_ids_path.exists() or not val_ids_path.exists()
        split_empty = (
            self.force_split
            or split_not_exists
            or len(pd.read_csv(train_ids_path)) == 0
            or len(pd.read_csv(test_ids_path)) == 0
            or len(pd.read_csv(val_ids_path)) == 0
        )
        if split_not_exists or split_empty:
            py_logger.info(f"Missing train, test or val ids from {self.split_path}")

            if "compound_dict" not in locals():
                py_logger.debug(f"Loading compound dictionary from {comp_path} ...")
                with open(comp_path) as handle:
                    compound_dict = json.load(handle)

            compound_list = list(compound_dict.keys())
            compound_list.sort()
            py_logger.debug(f"len(compound_list): {len(compound_list)}")
            py_logger.debug(f"Exemple compound: {compound_list[:2]}")

            py_logger.info("Creating the splitter...")
            self.splitter.set_compound_list(compound_list)

            py_logger.info(f"Creating the splits from {self.splitter}")
            split_ids = self.splitter.split()

            train_ids = split_ids["train"]
            test_ids = split_ids["test"]
            val_ids = split_ids["val"]

            py_logger.debug(f"len(train_ids): {len(train_ids)}")
            py_logger.debug(f"len(test_ids): {len(test_ids)}")
            py_logger.debug(f"len(val_ids): {len(val_ids)}")

            if len(train_ids) == 0 or len(test_ids) == 0 or len(val_ids) == 0:
                raise ValueError("One of the splits is empty.")

            py_logger.debug(
                f"Saving train, test and val ids to {train_ids_path}, {test_ids_path} and {val_ids_path} respectively ..."
            )

            train_ids_path.parent.mkdir(exist_ok=True)
            test_ids_path.parent.mkdir(exist_ok=True)
            val_ids_path.parent.mkdir(exist_ok=True)

            pd.Series(train_ids).to_csv(train_ids_path, index=False)
            pd.Series(test_ids).to_csv(test_ids_path, index=False)
            pd.Series(val_ids).to_csv(val_ids_path, index=False)

            if "retrieval" in split_ids:
                retrieval_ids = split_ids["retrieval"]
                py_logger.debug(f"len(retrieval_ids): {len(retrieval_ids)}")

                if len(retrieval_ids) > 0:
                    py_logger.debug(f"Saving retrieval ids to {retrieval_ids}")
                    retrieval_ids_path.parent.mkdir(exist_ok=True)
                    pd.Series(retrieval_ids).to_csv(retrieval_ids_path, index=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`,
        `self.test_dataset`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like
        random split twice!
        """

        if self.load_df is None:
            py_logger.info(f"Loading image metadata df from {self.image_metadata_path}")
            self.load_df = load_load_df_from_parquet(self.image_metadata_path)
            if self.data_root_dir is not None:
                for channel in self.channels:
                    self.load_df.loc[:, f"FileName_Orig{channel}"] = self.load_df[
                        f"FileName_Orig{channel}"
                    ].str.replace("/projects/", self.data_root_dir)

            self.image_list = self.load_df.index.tolist()
            self.n_images = len(self.image_list)

        if self.compound_dict is None:
            py_logger.info(f"Loading compound dictionary from {self.compound_metadata_path}")
            with open(self.compound_metadata_path) as handle:
                self.compound_dict = json.load(handle)
            self.compound_list = list(self.compound_dict.keys())
            self.n_compounds = len(self.compound_list)

        if self.train_cpds is None or self.val_cpds is None or self.test_cpds is None:
            py_logger.info(f"Loading train ids from {self.train_ids_path}")
            self.train_cpds = pd.read_csv(self.train_ids_path).iloc[:, 0].tolist()
            self.val_cpds = pd.read_csv(self.val_ids_path).iloc[:, 0].tolist()
            self.test_cpds = pd.read_csv(self.test_ids_path).iloc[:, 0].tolist()

            py_logger.info(
                f"Train, test, val lengths: {len(self.train_cpds)}, {len(self.test_cpds)}, {len(self.val_cpds)}"
            )

        if self.train_dataset is None:
            py_logger.info("Preparing train dataset")
            train_load_df = self.load_df[self.load_df[self.compound_col].isin(self.train_cpds)]
            train_compound_dict = {k: v for k, v in self.compound_dict.items() if k in self.train_cpds}
            self.train_dataset = self.dataset_cls(
                load_df=train_load_df,
                compound_dict=train_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.image_sampler,
                channels=self.channels,
                col_fstring=self.col_fstring,
                use_compond_cache=self.use_compond_cache,
            )

        if self.val_dataset is None:
            py_logger.info("Preparing validation dataset")
            val_load_df = self.load_df[self.load_df[self.compound_col].isin(self.val_cpds)]
            val_compound_dict = {k: v for k, v in self.compound_dict.items() if k in self.val_cpds}
            self.val_dataset = self.dataset_cls(
                load_df=val_load_df,
                compound_dict=val_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.image_sampler,
                channels=self.channels,
                col_fstring=self.col_fstring,
                use_compond_cache=self.use_compond_cache,
            )

        if stage == "test" and self.test_dataset is None:
            py_logger.info("Preparing test dataset")
            test_load_df = self.load_df[self.load_df[self.compound_col].isin(self.test_cpds)]
            test_compound_dict = {k: v for k, v in self.compound_dict.items() if k in self.test_cpds}
            self.test_dataset = self.dataset_cls(
                load_df=test_load_df,
                compound_dict=test_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.image_sampler,
                channels=self.channels,
                col_fstring=self.col_fstring,
                use_compond_cache=self.use_compond_cache,
            )

    def train_dataloader(self) -> DataLoader:
        train_kwargs = OmegaConf.to_container(self.dataloader_config.train, resolve=True)
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            **train_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        val_kwargs = OmegaConf.to_container(self.dataloader_config.val, resolve=True)
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collate_fn,
            **val_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        test_kwargs = OmegaConf.to_container(self.dataloader_config.test, resolve=True)
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collate_fn,
            **test_kwargs,
        )

    # def transfer_batch_to_device(self, batch: Any, device, dataloader_idx: int) -> Any:
    #     py_logger.debug("Transfer batch to device")
    #     new_batch = {k: v.to(device) for k, v in batch.items()}

    #     return new_batch

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
