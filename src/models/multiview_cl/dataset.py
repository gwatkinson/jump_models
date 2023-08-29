"""Module containing a torch Dataset that returns a molecule and an associated
image."""

import random

# import time
from typing import Callable, List, Optional

import pandas as pd
import torch
from PIL import UnidentifiedImageError
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
        transform: Optional[Callable] = None,
        compound_transform: Optional[Callable] = None,
        sampler: Optional[Callable[[List[str]], str]] = None,
        channels: List[str] = ("DNA", "AGP", "ER", "Mito", "RNA"),
        col_fstring: str = "FileName_Orig{channel}",
        max_tries: int = 10,
        use_compond_cache: bool = False,
    ):
        super().__init__()

        # data
        self.load_df = load_df

        # transforms
        self.transform = transform
        self.compound_transform = compound_transform

        # sampler
        self.max_tries = max_tries
        if sampler is None:
            self.sampler = random.choice
        else:
            self.sampler = sampler

        # string properties
        self.channels = channels
        self.col_fstring = col_fstring

        # caching
        self.use_compond_cache = use_compond_cache
        self.cached_compounds = {}

    def __len__(self):
        return self.n_compounds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={self.n_compounds}, n_images={self.n_images})"

    def transform_compound(self, compound):
        if self.cached_compounds and compound in self.cached_compounds:
            return self.cached_compounds[compound]
        else:
            tr_compound = self.compound_transform(compound)

            if self.use_compond_cache:
                self.cached_compounds[compound] = tr_compound
            return tr_compound

    def __getitem__(self, idx):
        # start = time.time()
        compound = self.compound_list[idx]  # An inchi or smiles string
        corresponding_images = self.compound_dict[compound]  # A list of indices into the load_df

        tries = 0
        fetched = False
        images_tried = []

        if self.compound_transform:
            tr_compound = self.transform_compound(compound)
        else:
            tr_compound = compound
        # ct_time = time.time()

        while not fetched and tries < self.max_tries and len(corresponding_images) > 0:
            try:
                # Random choice in the list of images
                image_id = self.sampler(corresponding_images)
                images_tried.append(image_id)

                image_paths = [
                    str(self.load_df.loc[image_id, self.col_fstring.format(channel=channel)])
                    for channel in self.channels
                ]
                # path_time = time.time()

                img_array = load_image_paths_to_array(image_paths)  # A numpy array: (5, 768, 768)
                img_array = torch.from_numpy(img_array)
                # img_time = time.time()

                if self.transform:
                    img_array = self.transform(img_array, image_id=image_id)
                # it_time = time.time()

                # py_logger.debug(
                #     f"Timing: compound_transform: {ct_time - start:.2f}s, path: {path_time - ct_time:.2f}s, img: {img_time - path_time:.2f}s, it: {it_time - img_time:.2f}s total: {it_time - start:.2f}s"
                # )

                return {"image": img_array, "compound": tr_compound}

            except UnidentifiedImageError:
                tries += 1
                py_logger.warning(f"Could not load image {image_id}. Try: {tries}/{self.max_tries}")
                corresponding_images = [i for i in corresponding_images if i != image_id]

        raise RuntimeError(f"Could not find an image for compound {compound} after {self.max_tries} tries.")
