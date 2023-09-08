"""Module containing a torch Dataset that returns a molecule and an associated
image."""

import random

# import time
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset

from src.utils import pylogger
from src.utils.io import load_image_paths_to_array

# from tqdm.rich import tqdm


py_logger = pylogger.get_pylogger(__name__)

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
        max_tries: int = 10,
        check_transform: bool = True,
        use_compond_cache: bool = False,
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
            max_tries (int, optional): The maximum number of tries to sample an image for a compound.
                Defaults to 10.
            use_compond_cache (bool, optional): Whether to cache the compound transforms. Defaults to True.
        """
        super().__init__()

        # transforms
        self.transform = transform
        self.compound_transform = compound_transform

        # data
        self.load_df = load_df
        self.compound_dict = compound_dict
        self.image_list = self.load_df.index.tolist()

        self.check_transform = check_transform
        # if check_transform:
        #     py_logger.info("Checking compounds with transformation...")
        #     bad_compounds = []
        #     for compound in tqdm(self.compound_dict):
        #         try:
        #             self.compound_transform(compound)
        #         except Exception as e:
        #             bad_compounds.append(compound)
        #             py_logger.warning(f"Could not transform compound {compound}. Error: {e}")

        bad_compounds = ["InChI=1S/Mo/q+6", "InChI=1S/3Na.V/q;;;+8"]

        for compound in bad_compounds:
            if compound in self.compound_dict:
                del self.compound_dict[compound]  # remove bad compounds from the dict

        self.compound_list = list(self.compound_dict.keys())

        # lenghts
        self.n_compounds = len(self.compound_list)
        self.n_images = len(self.image_list)

        # sampler
        self.max_tries = max_tries
        if sampler is None:
            self.sampler = random.choice
        else:
            self.sampler = sampler

        # string properties
        self.channels = channels
        self.col_fstring = col_fstring

        # # caching
        # self.use_compond_cache = use_compond_cache
        # self.cached_compounds = {}

    def __len__(self):
        return self.n_compounds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={self.n_compounds}, n_images={self.n_images})"

    def transform_compound(self, compound):
        # if self.cached_compounds and compound in self.cached_compounds:
        #     return self.cached_compounds[compound]
        # else:
        tr_compound = self.compound_transform(compound)

        # if self.use_compond_cache:
        #     self.cached_compounds[compound] = tr_compound
        return tr_compound

    def get_item(self, idx):
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
                    img_array = self.transform(img_array)
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

    def __getitem__(self, idx):
        tries = 0
        while tries < self.max_tries:
            try:
                out = self.get_item(idx)
                return out
            except Exception as e:
                idx = random.randint(0, self.n_compounds - 1)
                tries += 1
                py_logger.warning(f"Could not get item {idx}. Try: {tries}/{self.max_tries}. Error: {e}")
