"""Custom dataset for the OGB evaluation tasks."""

# flake8: noqa

import os
import os.path as osp
import shutil
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.modules.collate_fn import label_graph_collate_function
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


def smiles_str_to_list(smiles: str) -> List[str]:
    """"['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12',
    'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']"."""
    text = text[1:-1]
    lst = [i.strip()[1:-1] for i in text.split(",")]
    return lst


class HINTDataset(Dataset):
    def __init__(
        self,
        eval_df: pd.DataFrame,
        smiles_col: str = "smiless",
        target_col: str = "label",
        compound_transform: Optional[Callable] = None,
        use_cache: bool = False,
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
                Defaults to "smiles".
            compound_transform (Optional[Callable], optional):
                The compound transform to apply to the compounds.
                Defaults to None.
        """
        super().__init__()

        self.mapping = mapping
        self.targets = targets
        self.ids = ids
        self.smiles_col = smiles_col

        self.compound_transform = compound_transform

        self.cached_smiles = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.ids)

    def get_transformed_compound(self, compound):
        if self.use_cache and compound in self.cached_smiles:
            return self.cached_smiles[compound]
        else:
            tr_compound = self.compound_transform(compound)
            if self.use_cache:
                self.cached_smiles[compound] = tr_compound
            return tr_compound

    def __getitem__(self, idx: int):
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

        if self.compound_transform:
            tr_compound = self.get_transformed_compound(smile)
        else:
            tr_compound = smile

        y = self.mapping.loc[id_, self.targets].values.astype(float)
        y = torch.tensor(y)

        return {"compound": tr_compound, "label": y}

    def get_default_collate_fn(self):
        return label_graph_collate_function


class OGBBaseDataModule(LightningDataModule):
    """Base class for all OGB tasks."""

    dataset_name: str = "missing_dataset_name"
    dataset_url: str = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/"

    def __init__(
        self,
        root_dir: str,
        compound_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        targets: Optional[List[str]] = None,
        smiles_col: str = "smiles",
        split_type: Optional[str] = "scaffold",
        batch_size: int = 256,
        num_workers: int = 16,
        pin_memory: bool = False,
        prefetch_factor: int = 3,
        use_cache: bool = False,
    ):
        """Initializes the dataset.

        Args:
            root_dir (str):
                The root directory of the dataset.
                This should contain the different ogb datasets.
                Each dataset should be in a subfolder named after the dataset and contain the following files:
                    - mapping/mol.csv.gz
                    - split/{split_type}/<train|test|valid>.csv.gz
            compound_transform (Optional[Callable], optional):
                The compound transform to apply to the compounds.
                Defaults to None.
            collate_fn (Optional[Callable], optional):
                The collate function to use for the dataloader.
                Defaults to None.
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

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # dataloader args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        # targets
        self.compound_transform = compound_transform
        if self.compound_transform is not None:
            self.compound_transform.compound_str_type = "smiles"

        self.collate_fn = collate_fn
        self.got_default_collate_fn = False
        self.targets = targets
        self.smiles_col = smiles_col
        self.use_cache = use_cache

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
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
            logger.info(f"All files in the dataset folder {self.dataset_dir} not found.")

            if dir_exists:
                logger.warning(f"Deleting existing dataset folder {self.dataset_dir}.")
                shutil.rmtree(self.dataset_dir)

            logger.info(f"Creating dataset folder {self.dataset_dir}.")
            os.mkdir(self.dataset_dir)

            url = osp.join(self.dataset_url, f"{self.dataset_name}.zip")
            logger.info(f"Downloading dataset {self.dataset_name} from {url}.")
            download_and_extract_zip(url=url, path=self.root_dir)

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
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`,
        `self.test_dataset`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        if self.mapping is None:
            self.mapping = pd.read_csv(self.mapping_file)

        if self.targets is None:
            self.targets = self.mapping.columns.drop([self.smiles_col, "mol_id"]).tolist()

        if self.train_dataset is None:
            self.train_ids = pd.read_csv(self.train_ids_file, header=None).values.flatten().tolist()
            self.train_dataset = OGBDataset(
                mapping=self.mapping,
                targets=self.targets,
                ids=self.train_ids,
                smiles_col=self.smiles_col,
                compound_transform=self.compound_transform,
                use_cache=self.use_cache,
            )

        if self.collate_fn is None and not self.got_default_collate_fn:
            logger.info("Loading default collate function")
            self.collate_fn = self.train_dataset.get_default_collate_fn()
            self.got_default_collate_fn = True

        if self.val_dataset is None:
            self.val_ids = pd.read_csv(self.val_ids_file, header=None).values.flatten().tolist()
            self.val_dataset = OGBDataset(
                mapping=self.mapping,
                targets=self.targets,
                ids=self.val_ids,
                smiles_col=self.smiles_col,
                compound_transform=self.compound_transform,
                use_cache=self.use_cache,
            )

        if self.test_dataset is None:
            self.test_ids = pd.read_csv(self.test_ids_file, header=None).values.flatten().tolist()
            self.test_dataset = OGBDataset(
                mapping=self.mapping,
                targets=self.targets,
                ids=self.test_ids,
                smiles_col=self.smiles_col,
                compound_transform=self.compound_transform,
                use_cache=self.use_cache,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
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
