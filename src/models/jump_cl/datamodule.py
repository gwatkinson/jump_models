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
from src.modules.collate_fn import default_collate
from src.utils import pylogger
from src.utils.io import load_load_df_from_parquet, load_metadata_df_from_csv

py_logger = pylogger.get_pylogger(__name__)


class BasicJUMPDataModule(LightningDataModule):
    dataset_cls = MoleculeImageDataset  # The dataset class to use

    def __init__(
        self,
        image_metadata_path: str,
        compound_metadata_path: str,
        split_path: str,
        dataloader_config: DictConfig,
        train_ids_name: str = "train",
        force_split: bool = False,
        compound_col: str = "Metadata_InChI",
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = default_collate,
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
        self.image_sampler = image_sampler

        # other
        self.compound_col = compound_col
        self.image_metadata_path = image_metadata_path
        self.compound_metadata_path = compound_metadata_path
        self.use_compond_cache = use_compond_cache

        # split paths
        self.split_path = split_path
        self.force_split = force_split
        self.total_train_ids_path = osp.join(split_path, "total_train.csv")
        self.train_ids_name = train_ids_name
        self.train_ids_path = osp.join(split_path, f"{train_ids_name}.csv")
        self.val_ids_path = osp.join(split_path, "val.csv")
        self.test_ids_path = osp.join(split_path, "test.csv")
        self.retrieval_ids_path = osp.join(split_path, "retrieval.csv")

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
        total_train_ids_path = Path(self.total_train_ids_path)
        train_ids_path = Path(self.train_ids_path)
        test_ids_path = Path(self.test_ids_path)
        val_ids_path = Path(self.val_ids_path)
        retrieval_ids_path = Path(self.retrieval_ids_path)
        # load_dir = Path(self.local_load_data_dir)
        # meta_dir = Path(self.metadata_dir)
        # f_string = self.index_str
        # cols_to_keep = (
        #     self.id_cols
        #     + [self.col_fstring.format(channel=channel) for channel in self.channels]
        #     + [self.compound_col]
        #     + self.extra_cols
        # )

        # Prepare image metadata
        if not img_path.exists():
            self.prepare_load_df_with_meta()

            # py_logger.info("Preparing image metadata")
            # py_logger.debug(f"{img_path} does not exist.")
            # py_logger.debug(f"Loading load data df from {load_dir} ...")
            # load_df = load_load_df_from_parquet(load_dir)

            # py_logger.debug(f"Loading metadata df from {meta_dir} ...")
            # meta_df = load_metadata_df_from_csv(meta_dir)

            # py_logger.info("Merging metadata and load data...")
            # load_df_with_meta = load_df.merge(meta_df, how="left", on=self.id_cols).dropna(subset=[self.compound_col])
            # load_df_with_meta = load_df_with_meta.query("Metadata_PlateType == 'COMPOUND'")
            # load_df_with_meta["index"] = load_df_with_meta.apply(lambda x: f_string.format(**x), axis=1)

            # load_df_with_meta = load_df_with_meta.set_index("index", drop=True).loc[:, cols_to_keep]

            # py_logger.debug(f"load_df_with_meta ids unique: {load_df_with_meta.index.is_unique}")

            # py_logger.debug(f"Saving image metadata df to {img_path} ...")
            # img_path.parent.mkdir(exist_ok=True, parents=True)
            # load_df_with_meta.to_parquet(
            #     path=str(img_path),
            #     engine="pyarrow",
            #     compression="snappy",
            #     index=True,
            # )

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
        split_not_exists = not total_train_ids_path.exists() or not test_ids_path.exists() or not val_ids_path.exists()
        split_empty = (
            self.force_split
            or split_not_exists
            or len(pd.read_csv(total_train_ids_path)) == 0
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

            total_train_ids = split_ids["total_train"]
            test_ids = split_ids["test"]
            val_ids = split_ids["val"]

            py_logger.debug(f"len(total_train_ids): {len(total_train_ids)}")
            py_logger.debug(f"len(test_ids): {len(test_ids)}")
            py_logger.debug(f"len(val_ids): {len(val_ids)}")

            if len(total_train_ids) == 0 or len(test_ids) == 0 or len(val_ids) == 0:
                raise ValueError("One of the splits is empty.")

            py_logger.debug(
                f"Saving train, test and val ids to {total_train_ids_path}, {test_ids_path} and {val_ids_path} respectively ..."
            )

            total_train_ids_path.parent.mkdir(exist_ok=True)
            test_ids_path.parent.mkdir(exist_ok=True)
            val_ids_path.parent.mkdir(exist_ok=True)

            pd.Series(total_train_ids).to_csv(total_train_ids_path, index=False)
            pd.Series(test_ids).to_csv(test_ids_path, index=False)
            pd.Series(val_ids).to_csv(val_ids_path, index=False)

            if "retrieval" in split_ids:
                retrieval_ids = split_ids["retrieval"]
                py_logger.debug(f"len(retrieval_ids): {len(retrieval_ids)}")

                if len(retrieval_ids) > 0:
                    py_logger.debug(f"Saving retrieval ids to {retrieval_ids_path}")
                    retrieval_ids_path.parent.mkdir(exist_ok=True)
                    pd.Series(retrieval_ids).to_csv(retrieval_ids_path, index=False)

        if not train_ids_path.exists() or len(pd.read_csv(train_ids_path)) == 0:
            py_logger.info(f"Missing train ids from {self.split_path}")

            if "total_train_ids" not in locals():
                py_logger.debug(f"Loading total train ids from {total_train_ids_path} ...")
                total_train_ids = pd.read_csv(total_train_ids_path).iloc[:, 0].tolist()

            py_logger.info("Creating the train ids from the total train ids...")
            if self.splitter.compound_list is None:
                if "compound_dict" not in locals():
                    py_logger.debug(f"Loading compound dictionary from {comp_path} ...")
                    with open(comp_path) as handle:
                        compound_dict = json.load(handle)
                compound_list = list(compound_dict.keys())
                compound_list.sort()
                self.splitter.set_compound_list(compound_list)

            train_ids = self.splitter.split_train(total_train_ids)
            py_logger.debug(f"len(train_ids): {len(train_ids)}")

            if len(train_ids) == 0:
                raise ValueError("The train split is empty.")

            py_logger.debug(f"Saving train ids to {train_ids_path} ...")
            train_ids_path.parent.mkdir(exist_ok=True)
            pd.Series(train_ids).to_csv(train_ids_path, index=False)

        train_compound_path = Path(self.split_path) / f"{self.train_ids_name}_compound_dict.json"
        train_load_df_path = Path(self.split_path) / f"{self.train_ids_name}_load_df.parquet"
        if not train_compound_path.exists() or not train_load_df_path.exists():
            if "load_df_with_meta" not in locals():
                py_logger.debug(f"Loading local load data df from {img_path} ...")
                load_df_with_meta = load_load_df_from_parquet(img_path)

            if "compound_dict" not in locals():
                py_logger.debug(f"Loading compound dictionary from {comp_path} ...")
                with open(comp_path) as handle:
                    compound_dict = json.load(handle)

            train_cpds = pd.read_csv(train_ids_path).iloc[:, 0].tolist()

            train_compound_dict = {k: v for k, v in compound_dict.items() if k in train_cpds}
            train_ids = [item for sublist in train_compound_dict.values() for item in sublist]
            train_load_df = load_df_with_meta.loc[train_ids]

            with open(train_compound_path, "w") as handle:
                json.dump(train_compound_dict, handle)

            train_load_df.to_parquet(
                path=str(train_load_df_path),
                engine="pyarrow",
                compression="snappy",
                index=True,
            )

        val_compound_path = Path(self.split_path) / "val_compound_dict.json"
        val_load_df_path = Path(self.split_path) / "val_load_df.parquet"
        if not val_compound_path.exists() or not val_load_df_path.exists():
            if "load_df_with_meta" not in locals():
                py_logger.debug(f"Loading local load data df from {img_path} ...")
                load_df_with_meta = load_load_df_from_parquet(img_path)

            if "compound_dict" not in locals():
                py_logger.debug(f"Loading compound dictionary from {comp_path} ...")
                with open(comp_path) as handle:
                    compound_dict = json.load(handle)

            val_cpds = pd.read_csv(val_ids_path).iloc[:, 0].tolist()

            val_compound_dict = {k: v for k, v in compound_dict.items() if k in val_cpds}
            val_ids = [item for sublist in val_compound_dict.values() for item in sublist]
            val_load_df = load_df_with_meta.loc[val_ids]

            with open(val_compound_path, "w") as handle:
                json.dump(val_compound_dict, handle)

            val_load_df.to_parquet(
                path=str(val_load_df_path),
                engine="pyarrow",
                compression="snappy",
                index=True,
            )

        test_compound_path = Path(self.split_path) / "test_compound_dict.json"
        test_load_df_path = Path(self.split_path) / "test_load_df.parquet"
        if not test_compound_path.exists() or not test_load_df_path.exists():
            if "load_df_with_meta" not in locals():
                py_logger.debug(f"Loading local load data df from {img_path} ...")
                load_df_with_meta = load_load_df_from_parquet(img_path)

            if "compound_dict" not in locals():
                py_logger.debug(f"Loading compound dictionary from {comp_path} ...")
                with open(comp_path) as handle:
                    compound_dict = json.load(handle)

            test_cpds = pd.read_csv(test_ids_path).iloc[:, 0].tolist()

            test_compound_dict = {k: v for k, v in compound_dict.items() if k in test_cpds}
            test_ids = [item for sublist in test_compound_dict.values() for item in sublist]
            test_load_df = load_df_with_meta.loc[test_ids]

            with open(test_compound_path, "w") as handle:
                json.dump(test_compound_dict, handle)

            test_load_df.to_parquet(
                path=str(test_load_df_path),
                engine="pyarrow",
                compression="snappy",
                index=True,
            )

        retrieval_compound_path = Path(self.split_path) / "retrieval_compound_dict.json"
        retrieval_load_df_path = Path(self.split_path) / "retrieval_load_df.parquet"
        if not retrieval_compound_path.exists() or not retrieval_load_df_path.exists():
            if "load_df_with_meta" not in locals():
                py_logger.debug(f"Loading local load data df from {img_path} ...")
                load_df_with_meta = load_load_df_from_parquet(img_path)

            if "compound_dict" not in locals():
                py_logger.debug(f"Loading compound dictionary from {comp_path} ...")
                with open(comp_path) as handle:
                    compound_dict = json.load(handle)

            retrieval_cpds = pd.read_csv(retrieval_ids_path).iloc[:, 0].tolist()

            retrieval_compound_dict = {k: v for k, v in compound_dict.items() if k in retrieval_cpds}
            retrieval_ids = [item for sublist in retrieval_compound_dict.values() for item in sublist]
            retrieval_load_df = load_df_with_meta.loc[retrieval_ids]

            with open(retrieval_compound_path, "w") as handle:
                json.dump(retrieval_compound_dict, handle)

            retrieval_load_df.to_parquet(
                path=str(retrieval_load_df_path),
                engine="pyarrow",
                compression="snappy",
                index=True,
            )

    def replace_root_dir(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.data_root_dir is not None:
            for channel in self.channels:
                df.loc[:, f"FileName_Orig{channel}"] = df[f"FileName_Orig{channel}"].str.replace(
                    "/projects/", self.data_root_dir
                )
        return df

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`,
        `self.test_dataset`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        if self.train_dataset is None and (stage == "fit" or stage is None):
            py_logger.info("Preparing train dataset")

            train_load_df = pd.read_parquet(Path(self.split_path) / f"{self.train_ids_name}_load_df.parquet")
            train_load_df = self.replace_root_dir(train_load_df)

            with open(Path(self.split_path) / f"{self.train_ids_name}_compound_dict.json") as handle:
                train_compound_dict = json.load(handle)

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

        if self.val_dataset is None and (stage == "fit" or stage == "evaluate" or stage is None):
            py_logger.info("Preparing validation dataset")

            val_load_df = pd.read_parquet(Path(self.split_path) / "val_load_df.parquet")
            val_load_df = self.replace_root_dir(val_load_df)

            with open(Path(self.split_path) / "val_compound_dict.json") as handle:
                val_compound_dict = json.load(handle)

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

            test_load_df = pd.read_parquet(Path(self.split_path) / "test_load_df.parquet")
            test_load_df = self.replace_root_dir(test_load_df)

            with open(Path(self.split_path) / "test_compound_dict.json") as handle:
                test_compound_dict = json.load(handle)

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

        if stage == "retrieval" and self.retrieval_dataset is None:
            py_logger.info("Preparing retrieval dataset")

            retrieval_load_df = pd.read_parquet(Path(self.split_path) / "retrieval_load_df.parquet")
            retrieval_load_df = self.replace_root_dir(retrieval_load_df)

            with open(Path(self.split_path) / "retrieval_compound_dict.json") as handle:
                retrieval_compound_dict = json.load(handle)

            self.retrieval_dataset = self.dataset_cls(
                load_df=retrieval_load_df,
                compound_dict=retrieval_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.image_sampler,
                channels=self.channels,
                col_fstring=self.col_fstring,
                use_compond_cache=self.use_compond_cache,
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


class SingleSourceDataModule(BasicJUMPDataModule):
    source_to_keep = "source_1"

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
        id_cols = [_ for _ in self.id_cols if _ != "Metadata_Source"]

        py_logger.info("Preparing image metadata")
        py_logger.debug(f"{img_path} does not exist.")
        py_logger.debug(f"Loading load data df from {load_dir} ...")
        # !! This is the only change
        load_df = (
            load_load_df_from_parquet(load_dir)
            .query(f'Metadata_Source == "{self.source_to_keep}"')
            .drop(columns=["Metadata_Source"])
        )

        py_logger.debug(f"Loading metadata df from {meta_dir} ...")
        meta_df = load_metadata_df_from_csv(meta_dir)

        print(load_df.head().to_markdown())
        print(meta_df.head().to_markdown())

        py_logger.info("Merging metadata and load data...")
        load_df_with_meta = load_df.merge(meta_df, how="left", on=id_cols).dropna(subset=[self.compound_col])
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
