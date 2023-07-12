"""Custom dataset for the OGB evaluation tasks."""

import logging
import os.path as osp
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data_utils.utils import download_and_extract_zip

OGB_DATASETS = ["bbbp", "esol", "hiv", "lipophilicity", "tox21", "toxcast"]

logger = logging.getLogger(__name__)


class OGBDataset(Dataset):
    def __init__(
        self,
        mapping: pd.DataFrame,
        targets: List[str],
        ids: List[str],
        smiles_col: str = "smiles",
    ):
        """Initializes the dataset.

        Args:
            mapping (pd.DataFrame):
                The mapping dataframe.
            targets (List[str]):
                The list of target names.
            ids (List[str]):
                The list of ids.
            smiles_col (str, optional):
                The name of the column containing the smiles.
        """
        self.mapping = mapping
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """Returns the data at the given index.

        Args:
            idx (int):
                The index of the data to return.

        Returns:
            Tuple[str, torch.Tensor]:
                The smile and the classes.
        """
        id_ = self.ids[idx]

        smile = self.mapping.loc[id_, self.smiles_col]
        y = self.mapping.loc[id_, self.targets].values
        y = torch.tensor(y)

        return smile, y


class OGBDataModule(LightningDataModule):
    """Base class for all OGB tasks."""

    dataset_name: str = "missing_dataset_name"
    dataset_url: str = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/"

    def __init__(
        self,
        root_dir: str,
        targets: Optional[List[str]] = None,
        smiles_col: str = "smiles",
        split_type: Optional[str] = "scaffold",
        dataloader_config: Optional[DictConfig] = None,
    ):
        """Initializes the dataset.

        Args:
            root_dir (str):
                The root directory of the dataset.
                This should contain the different ogb datasets.
                Each dataset should be in a subfolder named after the dataset and contain the following files:
                    - mapping/mol.csv.gz
                    - split/{split_type}/<train|test|valid>.csv.gz
            targets (Optional[List[str]], optional):
                The list of target names in the dataset.
                If None, all columns except the smiles and mol_id columns are considered as targets.
                Defaults to None.
            smiles_col (str, optional):
                The name of the column containing the smiles.
                Defaults to "smiles".
            split_type (Optional[str], optional):
                The type of split to use. Can be either "scaffold" or "random" depending on the original split.
                Only used if dataset_folder is provided.
                Defaults to "scaffold".
            dataloader_config (Optional[DictConfig], optional):
                The configuration for the dataloader.
                Should contains the train, test, val keys with subkeys giving the dataloader parameters.
                Defaults to None.
        """

        if self.dataset_name not in OGB_DATASETS:
            logger.error(f"Dataset {self.dataset_name} not found in OGB datasets.")
            raise ValueError(f"Dataset {self.dataset_name} not found in OGB datasets.")

        super().__init__()

        self.save_hyperparameters(logger=False)

        # dataset paths
        self.root_dir = root_dir
        self.dataset_dir = osp.join(root_dir, self.dataset_name)
        self.mapping_file = osp.join(self.dataset_dir, "mapping/mol.csv.gz")
        self.train_ids_file = osp.join(self.dataset_dir, f"split/{split_type}/train.csv.gz")
        self.val_ids_file = osp.join(self.dataset_dir, f"split/{split_type}/valid.csv.gz")
        self.test_ids_file = osp.join(self.dataset_dir, f"split/{split_type}/test.csv.gz")

        # required attributes for the dataset loaded during setup
        self.mapping: Optional[pd.DataFrame] = None

        self.train_ids: Optional[List[str]] = None
        self.val_ids: Optional[List[str]] = None
        self.test_ids: Optional[List[str]] = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # dataloader args
        if dataloader_config is None:
            self.dataloader_train_args = {}
            self.dataloader_val_args = {}
            self.dataloader_test_args = {}
        else:
            self.dataloader_train_args = OmegaConf.to_container(dataloader_config.train, resolve=True)
            self.dataloader_val_args = OmegaConf.to_container(dataloader_config.val, resolve=True)
            self.dataloader_test_args = OmegaConf.to_container(dataloader_config.test, resolve=True)

        # targets
        self.targets = targets
        self.smiles_col = smiles_col

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        """Prepares the dataset."""
        if not osp.isdir(self.root_dir):
            logger.error(f"Dataset folder {self.root_dir} not found.")
            raise FileNotFoundError(f"Dataset folder {self.root_dir} not found.")

        dir_exists = osp.isdir(self.dataset_dir)
        all_file_exists = (
            osp.isfile(self.mapping_file)
            and osp.isfile(self.train_ids_file)
            and osp.isfile(self.val_ids_file)
            and osp.isfile(self.test_ids_file)
        )

        if not all_file_exists:
            logger.warning(f"All files in the dataset folder {self.dataset_dir} not found.")

            if dir_exists:
                logger.warning(f"Deleting existing dataset folder {self.dataset_dir}.")
                shutil.rmtree(self.dataset_dir)

            logger.info(f"Creating dataset folder {self.dataset_dir}.")
            osp.mkdir(self.dataset_dir)

            url = osp.join(self.dataset_url, f"{self.dataset_name}.zip")
            logger.info(f"Downloading dataset {self.dataset_name} from {url}.")
            download_and_extract_zip(url=url, path=self.dataset_dir)

            check_files = (
                osp.isfile(self.mapping_file)
                and osp.isfile(self.train_ids_file)
                and osp.isfile(self.val_ids_file)
                and osp.isfile(self.test_ids_file)
            )

            if not check_files:
                logger.error(f"Dataset files not found in {self.dataset_dir}.")
                raise FileNotFoundError(f"Dataset files not found in {self.dataset_dir}.")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        if self.mapping is None:
            self.mapping = pd.read_csv(self.mapping_file)

        if self.targets is None:
            self.targets = self.mapping.columns.drop([self.smiles_col, "mol_id"]).tolist()

        if (stage == "fit" or stage is None) and self.data_train is None:
            self.train_ids = pd.read_csv(self.train_ids_file, header=None).values.flatten().tolist()
            self.data_train = OGBDataset(
                mapping=self.mapping,
                targets=self.targets,
                ids=self.train_ids,
                smiles_col=self.smiles_col,
            )

        if stage == "validate" and self.data_val is None:
            self.val_ids = pd.read_csv(self.val_ids_file, header=None).values.flatten().tolist()
            self.data_val = OGBDataset(
                mapping=self.mapping,
                targets=self.targets,
                ids=self.val_ids,
                smiles_col=self.smiles_col,
            )

        if stage == "test" and self.data_test is None:
            self.test_ids = pd.read_csv(self.test_ids_file, header=None).values.flatten().tolist()
            self.data_test = OGBDataset(
                mapping=self.mapping,
                targets=self.targets,
                ids=self.test_ids,
                smiles_col=self.smiles_col,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            **self.dataloader_train_args,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            **self.dataloader_val_args,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            **self.dataloader_test_args,
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


class BBBPDataModule(OGBDataModule):
    dataset_name = "bbbp"


class EsolDataModule(OGBDataModule):
    dataset_name = "esol"


class HIVDataModule(OGBDataModule):
    dataset_name = "hiv"


class LipoDataModule(OGBDataModule):
    dataset_name = "lipophilicity"


class Tox21DataModule(OGBDataModule):
    dataset_name = "tox21"


class ToxCastDataModule(OGBDataModule):
    dataset_name = "toxcast"
