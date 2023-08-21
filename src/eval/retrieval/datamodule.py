import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.modules.collate_fn import default_collate, idr_flag_graph_collate_fn
from src.utils import pylogger
from src.utils.io import load_image_paths_to_array

logger = pylogger.get_pylogger(__name__)


class IDRRetrievalMoleculeDataset(Dataset):
    def __init__(
        self,
        selected_compounds: pd.DataFrame,
        compound_transform: Optional[Callable] = None,
        target_col: str = "Activity_Flag",
        smiles_col: str = "SMILES",
        use_cache: bool = True,
        gene: Optional[str] = None,
    ):
        super().__init__()

        self.selected_compounds = selected_compounds.sample(frac=1, random_state=42)
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.gene = gene

        self.target_to_num = {"A": 1, "N": 0}

        self.compound_transform = compound_transform

        self.use_cache = use_cache
        self.compound_cache = {}

    def __len__(self):
        return len(self.selected_compounds)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={len(self.selected_compounds)})"

    def get_transformed_compound(self, compound):
        if self.use_cache and compound in self.compound_cache:
            return self.compound_cache[compound]
        else:
            transformed_compound = self.compound_transform(compound)
            if self.use_cache:
                self.compound_cache[compound] = transformed_compound
            return transformed_compound

    def __getitem__(self, idx: int):
        row = self.selected_compounds.iloc[idx]
        target = self.target_to_num[row[self.target_col]]

        output = {"activity_flag": target}

        smile = row[self.smiles_col]
        if self.compound_transform:
            transformed_compound = self.get_transformed_compound(smile)
        else:
            transformed_compound = smile

        output["compound"] = transformed_compound

        return output

    def get_default_collate_fn(self):
        return idr_flag_graph_collate_fn


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
        selected_compounds_path: str,
        image_metadata_path: str,
        data_root_dir: str,
        image_batch_size: int = 256,
        compound_batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = False,
        prefetch_factor: int = 3,
        compound_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        compound_gene_col: str = "Gene_Symbol",
        image_gene_col: str = "Gene Symbol",
        col_fstring: str = "FileName_{channel}",
        channels: Optional[List[str]] = None,
        target_col: str = "Activity_Flag",
        smiles_col: str = "SMILES",
        use_cache: bool = True,
        mol_collate_fn: Optional[Callable] = default_collate,
        img_collate_fn: Optional[Callable] = default_collate,
    ):
        super().__init__()

        self.selected_compounds_path = selected_compounds_path
        self.image_metadata_path = image_metadata_path
        self.data_root_dir = data_root_dir

        self.transform = transform
        self.compound_transform = compound_transform
        if self.compound_transform is not None:
            self.compound_transform.compound_str_type = "smiles"

        self.mol_collate_fn = mol_collate_fn
        self.got_default_mol_collate_fn = False

        self.img_collate_fn = img_collate_fn
        self.got_default_img_collate_fn = False

        self.compound_gene_col = compound_gene_col
        self.image_gene_col = image_gene_col
        self.col_fstring = col_fstring
        self.channels = channels or ["DNA", "AGP", "ER", "Mito", "RNA"]
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.use_cache = use_cache

        # dataloader args
        self.image_batch_size = image_batch_size
        self.compound_batch_size = compound_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.selected_compounds: Optional[pd.DataFrame] = None
        self.image_metadata: Optional[pd.DataFrame] = None
        self.genes: Optional[List[str]] = None

        self.predict_molecule_datasets: Optional[List[Dataset]] = None
        self.predict_image_datasets: Optional[List[Dataset]] = None

    def prepare_data(self):  # ? Things to put here ?
        pass

    def setup(self, stage: Optional[str] = None, **kwargs):
        if self.selected_compounds is None:
            logger.info(f"Loading selected compounds from {self.selected_compounds_path}")
            self.selected_compounds = pd.read_csv(self.selected_compounds_path)

        if self.image_metadata is None:
            logger.info(f"Loading image metadata from {self.image_metadata_path}")
            self.image_metadata = pd.read_csv(self.image_metadata_path)

        if self.genes is None:
            self.genes = sorted(self.selected_compounds[self.compound_gene_col].unique().tolist())

        if stage in ["train", "validate", "test", None]:
            raise NotImplementedError

        if stage == "predict" and (self.predict_molecule_datasets is None or self.predict_image_datasets is None):
            mol_datasets = {}
            img_datasets = {}

            for gene in self.genes:
                sub_compounds = self.selected_compounds.query(f"`{self.compound_gene_col}` == '{gene}'")
                sub_image_metadata = self.image_metadata.query(f"`{self.image_gene_col}` == '{gene}'")

                mol_dataset = IDRRetrievalMoleculeDataset(
                    selected_compounds=sub_compounds,
                    compound_transform=self.compound_transform,
                    target_col=self.target_col,
                    smiles_col=self.smiles_col,
                    use_cache=self.use_cache,
                )
                mol_datasets[gene] = mol_dataset

                img_dataset = IDRRetrievalImageDataset(
                    image_metadata=sub_image_metadata,
                    data_root_dir=self.data_root_dir,
                    transform=self.transform,
                    col_fstring=self.col_fstring,
                    channels=self.channels,
                )
                img_datasets[gene] = img_dataset

                if self.mol_collate_fn is None and not self.got_default_mol_collate_fn:
                    logger.info("Loading default mol collate function")
                    self.mol_collate_fn = mol_dataset.get_default_collate_fn()
                    self.got_default_mol_collate_fn = True

                if self.img_collate_fn is None and not self.got_default_img_collate_fn:
                    logger.info("Loading default img collate function")
                    self.img_collate_fn = img_dataset.get_default_collate_fn()
                    self.got_default_img_collate_fn = True

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
                batch_size=self.compound_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                collate_fn=self.mol_collate_fn,
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
