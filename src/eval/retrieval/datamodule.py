import json
import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.modules.collate_fn import default_collate
from src.utils import pylogger
from src.utils.io import load_image_paths_to_array

logger = pylogger.get_pylogger(__name__)


class IDRRetrievalMoleculeDataset(Dataset):
    def __init__(
        self,
        compound_groups: List[Dict[str, List[int]]],
        excape_db: pd.DataFrame,
        smiles_col: str = "SMILES",
        target_col: str = "Activity_Flag",
        compound_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.compound_groups = compound_groups
        self.excape_db = excape_db
        self.smiles_col = smiles_col
        self.target_col = target_col

        self.compound_transform = compound_transform

        if hasattr(self.compound_transform, "compound_str_type"):
            self.compound_transform.compound_str_type = "smiles"

    def __len__(self):
        return len(self.selected_compounds)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_groups={len(self.compound_groups)})"

    def __getitem__(self, idx: int):
        indexes = self.compound_groups[idx]

        targets = self.excape_db.loc[indexes][self.target_col].values
        smiles = self.excape_db.loc[indexes][self.smiles_col].values

        transformed_compounds = []
        if self.compound_transform:
            for smile in smiles:
                transformed_compounds.append(self.compound_transform(smile))
            transformed_compounds = default_collate(transformed_compounds)
        else:
            transformed_compounds = smiles

        output = {
            "activity_flag": targets,  # List of 0s and 1s
            "compound": transformed_compounds,  # Batch of compounds (120 compounds)  -> use batch_size=1 and squeeze
        }

        return output


class IDRRetrievalImageDataset(Dataset):
    def __init__(
        self,
        image_metadata: pd.DataFrame,
        data_root_dir: str,
        transform: Optional[Callable] = None,
        col_fstring: str = "FileName_{channel}",
        channels: Optional[List[str]] = None,
        gene: Optional[str] = None,
    ):
        super().__init__()

        self.image_metadata = image_metadata.sample(frac=1, random_state=42)
        self.data_root_dir = data_root_dir
        self.col_fstring = col_fstring
        self.channels = channels or ["DNA", "AGP", "ER", "Mito", "RNA"]
        self.gene = gene

        self.transform = transform

    def __len__(self):
        return len(self.image_metadata)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_images={len(self.image_metadata)})"

    def __getitem__(self, idx: int):
        row = self.image_metadata.iloc[idx]

        image_paths = [
            osp.join(self.data_root_dir, str(row[self.col_fstring.format(channel=channel)]))
            for channel in self.channels
        ]

        img_array = load_image_paths_to_array(image_paths)
        img_array = torch.from_numpy(img_array)

        if self.transform:
            img_array = self.transform(img_array)

        return {"image": img_array}

    def get_default_collate_fn(self):
        return None


class IDRRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        image_metadata_path: str,
        excape_db_path: str,
        selected_group_path: str,
        data_root_dir: str,
        image_batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = False,
        prefetch_factor: int = 3,
        compound_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        image_gene_col: str = "Gene Symbol",
        col_fstring: str = "FileName_{channel}",
        channels: Optional[List[str]] = None,
        target_col: str = "Activity_Flag",
        smiles_col: str = "SMILES",
        img_collate_fn: Optional[Callable] = default_collate,
    ):
        super().__init__()

        # self.selected_compounds_path = selected_compounds_path
        self.image_metadata_path = image_metadata_path
        self.selected_group_path = selected_group_path
        self.excape_db_path = excape_db_path  # ../cpjump1/excape-db/processed_groups.json

        self.data_root_dir = data_root_dir

        self.transform = transform
        self.compound_transform = compound_transform
        if self.compound_transform is not None:
            self.compound_transform.compound_str_type = "smiles"

        self.got_default_mol_collate_fn = False

        self.img_collate_fn = img_collate_fn
        self.got_default_img_collate_fn = False

        self.image_gene_col = image_gene_col
        self.col_fstring = col_fstring
        self.channels = channels or ["DNA", "AGP", "ER", "Mito", "RNA"]
        self.target_col = target_col
        self.smiles_col = smiles_col

        # dataloader args
        self.image_batch_size = image_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.compound_groups: Optional[pd.DataFrame] = None
        self.excape_db: Optional[pd.DataFrame] = None
        # self.selected_compounds: Optional[pd.DataFrame] = None
        self.image_metadata: Optional[pd.DataFrame] = None
        self.genes: Optional[List[str]] = None

        self.predict_molecule_datasets: Optional[Dict[str, Dataset]] = None
        self.predict_image_datasets: Optional[Dict[str, Dataset]] = None

    def prepare_data(self):  # ? Things to put here ?
        pass

    def setup(self, stage: Optional[str] = None, **kwargs):
        if self.compound_groups is None:
            logger.info(f"Loading compound groups from {self.selected_group_path}")
            with open(self.selected_group_path) as f:
                self.compound_groups = json.load(f)

        if self.excape_db is None:
            logger.info(f"Loading excape db from {self.excape_db_path}")
            self.excape_db = pd.read_csv(self.excape_db_path)

        if self.image_metadata is None:
            logger.info(f"Loading image metadata from {self.image_metadata_path}")
            self.image_metadata = pd.read_csv(self.image_metadata_path)

        if self.genes is None:
            self.genes = sorted(self.compound_groups.keys())

        if stage in ["train", "validate", "test", None]:
            raise NotImplementedError

        if stage == "predict" and (self.predict_molecule_datasets is None or self.predict_image_datasets is None):
            mol_datasets = {}
            img_datasets = {}

            for gene in self.genes:
                groups = self.compound_groups[gene]
                sub_image_metadata = self.image_metadata.query(f"`{self.image_gene_col}` == '{gene}'")

                mol_datasets[gene] = IDRRetrievalMoleculeDataset(
                    compound_groups=groups,
                    excape_db=self.excape_db,
                    compound_transform=self.compound_transform,
                    target_col=self.target_col,
                    smiles_col=self.smiles_col,
                )

                img_datasets[gene] = IDRRetrievalImageDataset(
                    image_metadata=sub_image_metadata,
                    data_root_dir=self.data_root_dir,
                    transform=self.transform,
                    col_fstring=self.col_fstring,
                    channels=self.channels,
                )

            self.predict_molecule_datasets = mol_datasets
            self.predict_image_datasets = img_datasets

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self) -> Dict[str, Dict[str, DataLoader]]:
        out_dict = {}

        for gene in self.genes:
            mol_dataset = self.predict_molecule_datasets[gene]
            img_dataset = self.predict_image_datasets[gene]

            mol_dataloader = DataLoader(
                mol_dataset,
                batch_size=1,
            )

            img_dataloader = DataLoader(
                img_dataset,
                batch_size=self.image_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                collate_fn=self.img_collate_fn,
            )

            out_dict[gene] = {
                "molecule": mol_dataloader,
                "image": img_dataloader,
            }

        return out_dict

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or predict."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def __repr__(self):
        if self.selected_compounds is None or self.image_metadata is None:
            return f"{self.__class__.__name__}()"

        return f"""{self.__class__.__name__}(
    n_compounds={len(self.selected_compounds)},
    n_images={len(self.image_metadata)}
)"""
