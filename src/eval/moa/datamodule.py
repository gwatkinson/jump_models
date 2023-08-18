import os.path as osp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.modules.collate_fn import image_graph_label_collate_function, label_graph_collate_function
from src.splitters import BaseSplitter
from src.utils import pylogger
from src.utils.io import load_image_paths_to_array

logger = pylogger.get_pylogger(__name__)


class JumpMOADataset(Dataset):
    def __init__(
        self,
        moa_load_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        return_image: bool = True,
        return_compound: bool = False,
        target_col: str = "moa",
        smiles_col: str = "smiles",
        use_cache: bool = True,
        channels: Optional[List[str]] = None,
        data_root_dir: Optional[str] = None,
    ):
        """Initializes the dataset.

        Args:
            moa_load_df (pd.DataFrame):
                The load dataframe with the metadata.
            transform (Optional[Callable], optional):
                The transform to apply to the images.
            compound_transform (Optional[Callable], optional):
                The compound transform to apply to the compounds.
                Defaults to None.
            use_cache (bool, optional):
                Whether to use a cache for the compounds.
                Defaults to True.
            smiles_col (str, optional):
                The name of the column with the smiles.
                Defaults to "smiles".
            channels (Optional[List[str]], optional):
                The channels to use.
                Defaults to None.
            return_image (bool, optional):
                Whether to return the image.
                Defaults to True.
            return_compound (bool, optional):
                Whether to return the compound.
                Defaults to False.
        """
        super().__init__()

        self.moa_load_df = moa_load_df
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.channels = channels or ["DNA", "AGP", "ER", "Mito", "RNA"]
        self.return_image = return_image
        self.return_compound = return_compound
        self.data_root_dir = data_root_dir

        if data_root_dir:
            for channel in self.channels:
                self.moa_load_df.loc[:, f"FileName_Orig{channel}"] = self.moa_load_df[
                    f"FileName_Orig{channel}"
                ].str.replace("/projects/", data_root_dir)

        self.targets = self.moa_load_df[self.target_col].unique()
        self.targets.sort()
        self.target_to_num = {target: i for i, target in enumerate(self.targets)}
        self.n_targets = len(self.targets)

        self.transform = transform
        self.compound_transform = compound_transform

        self.n_compounds = moa_load_df[smiles_col].nunique()
        self.n_images = len(self.moa_load_df)

        self.use_cache = use_cache
        self.compound_cache = {}

    def __len__(self):
        return self.n_images

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={self.n_compounds}, n_images={self.n_images})"

    def get_transformed_compound(self, compound):
        if self.use_cache and compound in self.compound_cache:
            return self.compound_cache[compound]
        else:
            transformed_compound = self.compound_transform(compound)
            if self.use_cache:
                self.compound_cache[compound] = transformed_compound
            return transformed_compound

    def get_default_collate_fn(self):
        """Return the default collate function that should be used for this
        dataset."""
        if self.return_compound and self.return_image:
            return image_graph_label_collate_function
        elif not self.return_compound:
            return None
        elif not self.return_image:
            return label_graph_collate_function

    def __getitem__(self, idx: int):
        """Returns the data at the given index.

        Args:
            idx (int):
                The index of the data to return.

        Returns:
            Tuple[str, torch.Tensor]:
                The smile and the classes.
        """
        row = self.moa_load_df.iloc[idx]
        output = {"label": self.target_to_num[row[self.target_col]]}

        if self.return_image:
            img_paths = [row[f"FileName_Orig{channel}"] for channel in self.channels]

            img_array = load_image_paths_to_array(img_paths)  # A numpy array: (5, 768, 768)
            img_array = torch.from_numpy(img_array)

            if self.transform:
                img_array = self.transform(img_array)

            output["image"] = img_array

        if self.return_compound:
            smile = self.moa_load_df[self.smiles_col].iloc[idx]

            if self.compound_transform:
                transformed_compound = self.get_transformed_compound(smile)
            else:
                transformed_compound = smile

            output["compound"] = transformed_compound

        return output


class JumpMOADataModule(LightningDataModule):
    def __init__(
        self,
        moa_load_df_path: str,
        split_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        force_split: bool = False,
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        target_col: str = "moa",
        smiles_col: str = "smiles",
        return_image: bool = True,
        return_compound: bool = True,
        metadata_dir: Optional[str] = None,
        load_data_dir: Optional[str] = None,
        splitter: Optional[BaseSplitter] = None,
        max_obs_per_class: int = 1000,
        min_obs_per_class: int = 500,
        data_root_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.moa_load_df_path = moa_load_df_path
        self.split_path = split_path
        self.force_split = force_split

        self.transform = transform
        self.compound_transform = compound_transform
        if self.compound_transform is not None:
            self.compound_transform.compound_str_type = "smiles"

        self.collate_fn = collate_fn

        self.metadata_dir = metadata_dir
        self.load_data_dir = load_data_dir
        self.data_root_dir = data_root_dir

        self.train_path = Path(self.split_path) / "train.csv"
        self.val_path = Path(self.split_path) / "val.csv"
        self.test_path = Path(self.split_path) / "test.csv"

        self.splitter = splitter
        self.max_obs_per_class = max_obs_per_class
        self.min_obs_per_class = min_obs_per_class

        self.target_col = target_col
        self.smiles_col = smiles_col
        self.return_image = return_image
        self.return_compound = return_compound
        self.use_cache = kwargs.get("use_cache", False)

        self.got_default_collate_fn = False

        # dataloader args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.moa_load_df: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        moa_exists = Path(self.moa_load_df_path).exists()
        train_exist = self.train_path.exists()
        val_exist = self.val_path.exists()
        test_exist = self.test_path.exists()

        if not moa_exists:
            logger.info("Preparing MOA data")
            moa_path = osp.join(self.metadata_dir, "JUMP-MOA_compound_metadata.tsv")
            moa = pd.read_csv(moa_path, sep="\t", usecols=["smiles", "InChIKey", "moa", "pert_type"])

            meta = pd.read_csv(osp.join(self.metadata_dir, "local_metadata.csv"))

            load_df = pd.read_parquet(osp.join(self.load_data_dir, "final"))

            moa_wells = moa.dropna(subset=["InChIKey", "moa"]).merge(
                meta, left_on="InChIKey", right_on="Metadata_InChIKey", how="inner"
            )

            moa_load_df = load_df.merge(
                moa_wells, on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"], how="right"
            )

            # ? filter on other things ? (trt, platetype, etc.), equalize classes ?
            moa_load_df = moa_load_df.groupby("moa", as_index=False).apply(
                lambda x: x.sample(
                    min(
                        self.max_obs_per_class, (len(x) > self.min_obs_per_class) * len(x)
                    ),  # 0 if len(x) <= min_obs_per_class else len(x)
                    replace=False,
                )
            )

            logger.info(f"Saving MOA data to {self.moa_load_df_path} ({len(moa_load_df)} rows)")
            Path(self.moa_load_df_path).parent.mkdir(parents=True, exist_ok=True)
            moa_load_df.to_csv(self.moa_load_df_path, index=False)

        if not train_exist or not val_exist or not test_exist or self.force_split:
            logger.info("Splitting MOA")
            moa_load_df = pd.read_csv(self.moa_load_df_path)

            y = moa_load_df[self.target_col].values

            self.splitter.set_compound_list(y)
            split_ids = self.splitter.split()

            for phase in ["train", "val", "test"]:
                save_path = Path(self.split_path) / f"{phase}.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                pd.Series(split_ids[phase]).to_csv(save_path, index=False)

    def setup(self, stage: Optional[str] = None, **kwargs):
        if self.moa_load_df is None:
            logger.info(f"Loading MOA data from {self.moa_load_df_path}")
            self.moa_load_df = pd.read_csv(self.moa_load_df_path)
            self.num_classes = self.moa_load_df[self.target_col].nunique()

        if stage == "test" and self.test_dataset is None:
            test_ids = pd.read_csv(self.test_path, header=None).values.flatten().tolist()
            test_df = self.moa_load_df.iloc[test_ids]

            logger.info(f"Creating test dataset ({len(test_df)} rows)")
            self.test_dataset = JumpMOADataset(
                moa_load_df=test_df,
                transform=self.transform,
                compound_transform=self.compound_transform,
                return_image=self.return_image,
                return_compound=self.return_compound,
                target_col=self.target_col,
                smiles_col=self.smiles_col,
                use_cache=self.use_cache,
                data_root_dir=self.data_root_dir,
                **kwargs,
            )

        elif self.train_dataset is None or self.val_dataset is None:
            train_ids = pd.read_csv(self.train_path, header=None).values.flatten().tolist()
            val_ids = pd.read_csv(self.val_path, header=None).values.flatten().tolist()
            train_df = self.moa_load_df.iloc[train_ids]
            val_df = self.moa_load_df.iloc[val_ids]

            logger.info(f"Creating train and val datasets ({len(train_df)} and {len(val_df)} rows)")
            self.train_dataset = JumpMOADataset(
                moa_load_df=train_df,
                transform=self.transform,
                compound_transform=self.compound_transform,
                return_image=self.return_image,
                return_compound=self.return_compound,
                target_col=self.target_col,
                smiles_col=self.smiles_col,
                use_cache=self.use_cache,
                data_root_dir=self.data_root_dir,
                **kwargs,
            )

            self.val_dataset = JumpMOADataset(
                moa_load_df=val_df,
                transform=self.transform,
                compound_transform=self.compound_transform,
                return_image=self.return_image,
                return_compound=self.return_compound,
                target_col=self.target_col,
                smiles_col=self.smiles_col,
                use_cache=self.use_cache,
                data_root_dir=self.data_root_dir,
                **kwargs,
            )

            if self.collate_fn is None and not self.got_default_collate_fn:
                logger.info("Loading default collate function")
                self.collate_fn = self.train_dataset.get_default_collate_fn()
                self.got_default_collate_fn = True

    def train_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
            **kwargs,
        )

    def val_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            **kwargs,
        )

    def test_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            **kwargs,
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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.moa_load_df_path})"
