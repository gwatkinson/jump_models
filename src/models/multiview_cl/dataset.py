"""Module containing a torch Dataset that returns a molecule and an associated
image."""

import random
from collections import defaultdict

# import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import pylogger
from src.utils.io import load_image_paths_to_array

py_logger = pylogger.get_pylogger(__name__)


class MultiviewDataset(Dataset):
    """This Dataset returns both a molecule, in the SMILES format and an
    associated image."""

    def __init__(
        self,
        load_df: pd.DataFrame,
        compound_dict: Dict[str, List[str]],
        n_views: int = 3,
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        channels: List[str] = ("DNA", "AGP", "ER", "Mito", "RNA"),
        col_fstring: str = "FileName_Orig{channel}",
        max_tries: int = 10,
        **kwargs,
    ):
        super().__init__()

        # data
        self.load_df = load_df  # A dataframe containing the paths to the images on the disk.
        self.compound_dict = compound_dict  # A dictionary mapping a compound to a list of indices into the load_df

        bad_compounds = ["InChI=1S/Mo/q+6", "InChI=1S/3Na.V/q;;;+8"]

        for compound in bad_compounds:
            if compound in self.compound_dict:
                del self.compound_dict[compound]  # remove bad compounds from the dict

        self.compound_list = list(self.compound_dict.keys())  # List of all compounds

        self.n_compounds = len(self.compound_list)  # Number of compounds
        self.n_images = len(load_df)  # Number of images
        self.n_views = n_views  # Number of views per compound

        # transforms
        self.transform = transform
        self.compound_transform = compound_transform

        self.max_tries = max_tries

        # string properties
        self.channels = channels
        self.col_fstring = col_fstring

    def replace_root_dir(self, new_root_dir: str, old_root_dir: str = "/projects/"):
        for channel in self.channels:
            col = self.col_fstring.format(channel=channel)
            self.load_df[col] = self.load_df[col].str.replace(old_root_dir, new_root_dir)

    def __len__(self):
        return self.n_compounds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={self.n_compounds}, n_images={self.n_images})"

    def get_item(self, idx):
        # Get the compound
        compound = self.compound_list[idx]  # An inchi or smiles string

        if self.compound_transform:
            tr_compound = self.compound_transform(compound)  # A graph or feature vector
        else:
            tr_compound = compound

        # Get the images
        corresponding_images = self.compound_dict[compound]  # A list of indices into the load_df
        corresponding_batches = [
            "__".join(img.split("__")[:2]) for img in corresponding_images
        ]  # The batch of the images
        unique_batches = list(set(corresponding_batches))
        batch_to_num = {batch: i for i, batch in enumerate(unique_batches)}  # A dictionary mapping a batch to a number
        num_batches = len(unique_batches)

        batch_to_images = defaultdict(list)
        for img, batch in zip(corresponding_images, corresponding_batches):
            batch_to_images[batch].append(img)  # A dictionary mapping a batch to a list of images in that batch

        if num_batches >= self.n_views:  # We have enough batches to sample from
            batch_to_try = np.random.choice(unique_batches, size=self.n_views, replace=False)
        else:  # We need to sample with replacement
            batch_to_try = np.concatenate(
                [
                    np.random.permutation(unique_batches),
                    np.random.choice(unique_batches, size=self.n_views - num_batches, replace=True),
                ]
            )

        images = []  # n_views * (5, 768, 768)
        loaded_images = []

        for batch in batch_to_try:
            batch_images = [
                imgs for imgs in batch_to_images[batch] if imgs not in loaded_images
            ]  # A list of images in the batch that have not been loaded yet
            tries = 0
            fetched = False

            while not fetched and tries < self.max_tries and len(batch_images) > 0:
                try:
                    # Random choice in the list of images
                    image_id = random.choice(batch_images)
                    loaded_images.append(image_id)

                    image_paths = [
                        str(self.load_df.loc[image_id, self.col_fstring.format(channel=channel)])
                        for channel in self.channels
                    ]

                    img_array = load_image_paths_to_array(image_paths)  # A numpy array: (5, 768, 768)
                    # img_array = torch.from_numpy(img_array)

                    if self.transform:
                        img_array = self.transform(img_array)

                    images.append(img_array)
                    fetched = True

                except Exception as e:
                    tries += 1
                    py_logger.warning(f"Could not load image {e}. Try: {tries}/{self.max_tries}")
                    if "image_id" in locals():
                        batch_images = [i for i in batch_images if i != image_id]

            if not fetched:
                raise RuntimeError(f"Could not find an image for compound {compound} after {self.max_tries} tries.")

        images = torch.stack(images, dim=0)  # (n_views, 5, 768, 768)
        batches = [batch_to_num[batch] for batch in batch_to_try]

        return {"compound": tr_compound, "image": images, "batch": batches, "compound_name": compound}

    def __getitem__(self, idx):
        tries = 0
        while tries < self.max_tries:
            try:
                out = self.get_item(idx)
                return out
            except Exception as e:
                idx = random.randint(0, self.n_compounds - 1)
                tries += 1
                py_logger.warning(
                    f"Could not get item {idx}. Trying random compound. Try: {tries}/{self.max_tries}. Error: {e}"
                )
        raise RuntimeError(f"Could not get item {idx} after {self.max_tries} tries.")
