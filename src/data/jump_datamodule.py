"""Module containing the DataModules using the JUMP dataset."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data.image_smiles_dataset import MoleculeImageDataset
from src.data.splits import RandomSplitter, ScaffoldSplitter
from src.data_utils.utils import load_load_df_from_parquet, load_metadata_df_from_csv

py_logger = logging.getLogger(__name__)


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
        train_ids_path: Optional[Union[str, Path]],
        test_ids_path: Optional[Union[str, Path]],
        val_ids_path: Optional[Union[str, Path]],
        dataloader_config: DictConfig,
        compound_col: str = "Metadata_InChI",
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        image_sampler: Optional[Callable[[List[str]], str]] = None,
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
            compound_col (str): Name of the column containing the compound names in the image metadata df.
            train_ids_path (Optional[Union[str, Path]]): Path to the train ids csv file.
            test_ids_path (Optional[Union[str, Path]]): Path to the test ids csv file.
            val_ids_path (Optional[Union[str, Path]]): Path to the val ids csv file.
            dataloader_config (DictConfig): Config dict for the dataloaders.
                Should contains the train, test, val keys with subkeys giving the dataloader parameters.
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
        self.save_hyperparameters(logger=False)

        # metadata
        self.load_df: Optional[pd.DataFrame] = None
        self.compound_dict: Optional[Dict[str, List[str]]] = None

        # split ids
        self.train_cpds: Optional[List] = None
        self.test_cpds: Optional[List] = None
        self.val_cpds: Optional[List] = None

        # data transformations
        self.transform = transform
        self.compound_transform = compound_transform

        # datasets
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # data loaders
        self.dataloader_config = dataloader_config

        # other
        self.compound_col = compound_col

        # kwargs
        self.index_str = kwargs.get(
            "index_str", "{Metadata_Source}__{Metadata_Batch}__{Metadata_Plate}__{Metadata_Well}__{Metadata_Site}"
        )
        self.metadata_dir = kwargs.get("metadata_dir", "../cpjump1/jump/metadata/complete_metadata.csv")
        self.local_load_data_dir = kwargs.get("local_load_data_dir", "../cpjump1/jump/load_data/final/")
        self.splitter = kwargs.get("splitter", RandomSplitter)
        self.channels = kwargs.get("channels", ["DNA", "AGP", "ER", "Mito", "RNA"])
        self.col_fstring = kwargs.get("col_fstring", "FileName_Orig{channel}")
        self.id_cols = kwargs.get("id_cols", ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
        self.extra_cols = kwargs.get("extra_cols", ["Metadata_PlateType", "Metadata_Site"])

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        img_path = Path(self.hparams.image_metadata_path)
        comp_path = Path(self.hparams.compound_metadata_path)
        train_ids_path = Path(self.hparams.train_ids_path)
        test_ids_path = Path(self.hparams.test_ids_path)
        val_ids_path = Path(self.hparams.val_ids_path)
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
            py_logger.info("=== Preparing image metadata ===")
            py_logger.info(f"{img_path} does not exist.")
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
            py_logger.info("=== Preparing compound metadata ===")
            py_logger.info(f"{comp_path} does not exist.")

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
        if not train_ids_path.exists() or not test_ids_path.exists() or not val_ids_path.exists():
            # ! To test
            py_logger.info("=== Missing train, test or val ids ===")

            if "compound_dict" not in locals():
                py_logger.debug(f"Loading compound dictionary from {comp_path} ...")
                with open(comp_path) as handle:
                    compound_dict = json.load(handle)

            compound_list = list(compound_dict.keys())
            compound_list.sort()

            py_logger.info("Creating the splitter...")
            self.splitter.set_compound_list(compound_list)
            py_logger.info(f"Creating them from {self.splitter}")

            train_ids, test_ids, val_ids = self.splitter.split()

            py_logger.debug(
                f"Saving train, test and val ids to {train_ids_path}, {test_ids_path} and {val_ids_path} respectively ..."
            )

            train_ids_path.parent.mkdir(exist_ok=True)
            test_ids_path.parent.mkdir(exist_ok=True)
            val_ids_path.parent.mkdir(exist_ok=True)

            pd.Series(train_ids).to_csv(train_ids_path, index=False)
            pd.Series(test_ids).to_csv(test_ids_path, index=False)
            pd.Series(val_ids).to_csv(val_ids_path, index=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        if self.load_df is None:
            py_logger.debug(f"Loading image metadata df from {self.hparams.image_metadata_path}")
            self.load_df = load_load_df_from_parquet(self.hparams.image_metadata_path)
            self.image_list = self.load_df.index.tolist()
            self.n_images = len(self.image_list)

        if self.compound_dict is None:
            py_logger.debug(f"Loading compound dictionary from {self.hparams.compound_metadata_path}")
            with open(self.hparams.compound_metadata_path) as handle:
                self.compound_dict = json.load(handle)
            self.compound_list = list(self.compound_dict.keys())
            self.n_compounds = len(self.compound_list)

        if self.train_cpds is None or self.val_cpds is None or self.test_cpds is None:
            py_logger.debug(
                f"Loading train ids from {self.hparams.train_ids_path}, val ids from {self.hparams.val_ids_path} and test ids from {self.hparams.test_ids_path}"
            )
            self.train_cpds = pd.read_csv(self.hparams.train_ids_path, header=None)[0].tolist()
            self.val_cpds = pd.read_csv(self.hparams.val_ids_path, header=None)[0].tolist()
            self.test_cpds = pd.read_csv(self.hparams.test_ids_path, header=None)[0].tolist()

        if stage == "fit" and self.data_train is None:
            py_logger.debug("=== Preparing train dataset ===")
            train_load_df = self.load_df[self.load_df[self.compound_col].isin(self.train_cpds)]
            train_compound_dict = {k: v for k, v in self.compound_dict.items() if k in self.train_cpds}
            self.data_train = self.dataset_cls(
                load_df=train_load_df,
                compound_dict=train_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.hparams.image_sampler,
                channels=self.channels,
                col_fstring=self.col_fstring,
            )

        if stage == "validate" and self.data_val is None:
            val_load_df = self.load_df[self.load_df[self.compound_col].isin(self.val_cpds)]
            val_compound_dict = {k: v for k, v in self.compound_dict.items() if k in self.val_cpds}
            self.data_val = self.dataset_cls(
                load_df=val_load_df,
                compound_dict=val_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.hparams.image_sampler,
                channels=self.channels,
                col_fstring=self.col_fstring,
            )

        if stage == "test" and self.data_test is None:
            test_load_df = self.load_df[self.load_df[self.compound_col].isin(self.test_cpds)]
            test_compound_dict = {k: v for k, v in self.compound_dict.items() if k in self.test_cpds}
            self.data_test = self.dataset_cls(
                load_df=test_load_df,
                compound_dict=test_compound_dict,
                transform=self.transform,
                compound_transform=self.compound_transform,
                sampler=self.hparams.image_sampler,
                channels=self.channels,
                col_fstring=self.col_fstring,
            )

    def train_dataloader(self) -> DataLoader:
        train_kwargs = OmegaConf.to_container(self.dataloader_config.train, resolve=True)
        return DataLoader(
            dataset=self.data_train,
            **train_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        val_kwargs = OmegaConf.to_container(self.dataloader_config.val, resolve=True)
        return DataLoader(
            dataset=self.data_val,
            **val_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        test_kwargs = OmegaConf.to_container(self.dataloader_config.test, resolve=True)
        return DataLoader(
            dataset=self.data_test,
            **test_kwargs,
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


if __name__ == "__main__":
    from datamol import from_inchi, to_smiles

    def compound_transform(x: str) -> str:
        return to_smiles(from_inchi(x))

    # file_handler = logging.FileHandler(filename="./logs/tmp.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [stdout_handler]
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    py_logger.info("Creating data module")

    dm = BasicJUMPDataModule(
        image_metadata_path="/projects/cpjump1/jump/models/metadata/images_metadata.parquet",
        compound_metadata_path="/projects/cpjump1/jump/models/metadata/compound_dict.json",
        compound_col="Metadata_InChI",
        train_ids_path="/projects/cpjump1/jump/models/splits/small_test/train_ids.csv",
        test_ids_path="/projects/cpjump1/jump/models/splits/small_test/test_ids.csv",
        val_ids_path="/projects/cpjump1/jump/models/splits/small_test/val_ids.csv",
        dataloader_config=DictConfig(
            {
                "train": DictConfig(
                    {
                        "batch_size": 32,
                        "num_workers": 0,
                        "pin_memory": False,
                        "shuffle": True,
                    }
                ),
                "val": DictConfig(
                    {
                        "batch_size": 32,
                        "num_workers": 0,
                        "pin_memory": False,
                        "shuffle": False,
                    }
                ),
                "test": DictConfig(
                    {
                        "batch_size": 32,
                        "num_workers": 0,
                        "pin_memory": False,
                        "shuffle": False,
                    }
                ),
            }
        ),
        transform=None,
        compound_transform=compound_transform,
        image_sampler=None,
        metadata_dir="/projects/cpjump1/jump/metadata/complete_metadata.csv",
        local_load_data_dir="/projects/cpjump1/jump/load_data/final/",
        splitter=ScaffoldSplitter(
            train=800,
            test=200,
            val=100,
        ),
        index_str="{Metadata_Source}__{Metadata_Batch}__{Metadata_Plate}__{Metadata_Well}__{Metadata_Site}",
        channels=["DNA", "AGP", "ER", "Mito", "RNA"],
        col_fstring="FileName_Orig{channel}",
        id_cols=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
        extra_cols=["Metadata_PlateType", "Metadata_Site"],
    )

    py_logger.info("Preparing data")

    dm.prepare_data()

    py_logger.info("Setting up data")

    dm.setup(stage="fit")

    py_logger.info(f"Testing dataset: {dm.data_train}")
    py_logger.debug(f"Dataset obs: {dm.data_train[0]}")

    py_logger.info("Getting a batch")
    py_logger.debug(f"Test dataloader: {dm.train_dataloader()}")

    print("len(train_cpds): ", len(dm.train_cpds))
    print("len(test_cpds): ", len(dm.test_cpds))
    print("len(val_cpds): ", len(dm.val_cpds))

    for batch in dm.train_dataloader():
        print(batch)
        print(batch["image"].shape)
        print(batch["compound"])
        break
