"""Module containing a torch Dataset that returns a molecule and an associated
image."""

import logging
import random
import time
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.io import load_image_paths_to_array

py_logger = logging.getLogger(__name__)

default_channels = ["DNA", "AGP", "ER", "Mito", "RNA"]


class MoleculeImageDataset(Dataset):
    """This Dataset returns both a molecule, in the SMILES format and an
    associated image."""

    def __init__(
        self,
        load_df: pd.DataFrame,
        compound_dict: Dict[str, List[str]],
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        sampler: Optional[Callable[[List[str]], str]] = None,
        channels: List[str] = default_channels,
        col_fstring: str = "FileName_Orig{channel}",
    ):
        """Initializes the dataset.

        Args:
            load_df (pd.DataFrame): A dataframe containing the paths to the images.
            compound_dict (Dict[str, List[str]]): A dictionary mapping a compound
                to a list of indices into the load_df.
            transform (Optional[Callable], optional): A transform to apply to the
                image. Usually torchvision transforms. Defaults to None.
            compound_transform (Optional[Callable], optional): A transform to apply
                to the compound. This can be a tokenizer or a featurizer transforming a string into a Graph.
                Defaults to None.
            sampler (Optional[Callable[[List[str]], str]], optional): A function that
                samples an index from a list of indices. Defaults to None.
            channels (List[str], optional): A list of strings that are used to
                format the column names in the load_df. Defaults to default_channels.
            col_fstring (str, optional): A format string that is used to format the
                column names in the load_df. Defaults to "FileName_Orig{channel}".
        """

        # data
        self.load_df = load_df
        self.compound_dict = compound_dict
        self.compound_list = list(self.compound_dict.keys())
        self.n_compounds = len(self.compound_list)
        self.image_list = self.load_df.index.tolist()
        self.n_images = len(self.image_list)

        # transforms
        self.transform = transform
        self.compound_transform = compound_transform

        # sampler
        if sampler is None:
            self.sampler = random.choice
        else:
            self.sampler = sampler

        # string properties
        self.channels = channels
        self.col_fstring = col_fstring

        # caching
        self.cached_compound = {}

    def __len__(self):
        return self.n_compounds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={self.n_compounds}, n_images={self.n_images})"

    def transform_compound(self, compound):
        if compound in self.cached_compound:
            return self.cached_compound[compound]
        else:
            tr_compound = self.compound_transform(compound)
            self.cached_compound[compound] = tr_compound
            return tr_compound

    def __getitem__(self, idx):
        start = time.time()
        compound = self.compound_list[idx]  # An inchi or smiles string
        image_id = self.sampler(self.compound_dict[compound])  # An index into the load_df
        image_paths = [
            str(self.load_df.loc[image_id, self.col_fstring.format(channel=channel)]) for channel in self.channels
        ]
        path_time = time.time()

        img_array = load_image_paths_to_array(image_paths)  # A numpy array: (5, 768, 768)
        img_array = torch.from_numpy(img_array)
        img_time = time.time()

        if self.transform:
            img_array = self.transform(img_array)
        it_time = time.time()

        if self.compound_transform:
            tr_compound = self.transform_compound(compound)
        else:
            tr_compound = compound
        ct_time = time.time()

        py_logger.debug(
            f"Timing: path: {path_time - start:.2f}s, img: {img_time - path_time:.2f}s, it: {it_time - img_time:.2f}s, ct: {ct_time - it_time:.2f}s, total: {ct_time - start:.2f}s"
        )

        return {"image": img_array, "compound": tr_compound}
