import random

# import time
from typing import Callable, Dict, List, Optional, Literal, Sequence

import pandas as pd
import torch
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset
from tqdm.rich import tqdm

from src.utils import pylogger
from src.utils.io import load_image_paths_to_array

# from tqdm.rich import tqdm


py_logger = pylogger.get_pylogger(__name__)

default_channels = ["DNA", "AGP", "ER", "Mito", "RNA"]
bad_compounds = ["InChI=1S/Mo/q+6", "InChI=1S/3Na.V/q;;;+8"]

class MoleculeDataset(Dataset):
    def __init__(
        self,
        compound_list: List[str],
        compound_to_exclude: Optional[List[str]] = bad_compounds,
        compound_str_type: Optional[Literal["smiles", "inchi", "selfies"]] = None,
        compound_transform: Optional[Callable] = None,
        check_compound_transform: bool = False,
        use_compond_cache: bool = False,
        max_tries: int = 10,
        verbose: bool = False,
    ):
        """Initializes the dataset.

        Args:
            compound_list (List[str]): A list of compounds in the compound_str_type format.
            compound_to_exclude (Optional[List[str]], optional): A list of compounds to exclude.
            compound_str_type (Literal["smiles", "inchi", "selfies"]): The type of the compound strings.
            compound_transform (Optional[Callable], optional): A transform to apply
                to the compound. This can be a tokenizer or a featurizer transforming a string into a Graph.
                Defaults to None.
            check_compound_transform (bool, optional): Whether to check if the compound_transform works for all compounds.
                Defaults to False.
            use_compond_cache (bool, optional): Whether to cache the compound transforms. Defaults to False.
            max_tries (int, optional): The maximum number of tries to get a compound.
                If there is an error during the transformation of the compound, another compound is tried.
                This may lead to the same compound being returned multiple times.
            verbose (bool, optional): Whether to show a progress bar when checking the compounds. Defaults to False.
        
        This is a base class for datasets that only return a compound.
        But it can also be used for more complex datasets that return a compound and an image.
        """
        super().__init__()
        self.compound_list = compound_list
        self.compound_to_exclude = compound_to_exclude
        self.compound_str_type = compound_str_type
        self.compound_transform = compound_transform
        self.check_compound_transform = check_compound_transform
        self.use_compond_cache = use_compond_cache
        self.max_tries = max_tries
        self.verbose = verbose
        
        self.n_compounds = len(self.compound_list)
        
        if compound_str_type and hasattr(self.compound_transform, "compound_str_type"):
            self.compound_transform.compound_str_type = self.compound_str_type
        
        if self.compound_to_exclude:
            self.remove_bad_compounds()
        
        if self.check_compound_transform:
            self.verify_compounds()
        
        if self.use_compond_cache:
            self.setup_cache()
    
    def remove_bad_compounds(self):
        for compound in self.bad_compounds:
            if compound in self.compound_dict:
                del self.compound_dict[compound]  # remove bad compounds from the dict
    
    def verify_compounds(self):
        if self.verbose:
            py_logger.info("Checking compounds with transformation...")
            
        bad_compounds = []
        pbar = tqdm(self.compound_list) if self.verbose else self.compound_list
        for compound in pbar:
            try:
                self.compound_transform(compound)
            except Exception as e:
                bad_compounds.append(compound)
                if self.verbose:
                    py_logger.warning(f"Could not transform compound {compound}. Error: {e}")

        for compound in bad_compounds:
            self.compound_list.remove(compound)
            
    def setup_cache(self):
        self.cached_compounds = {}

    def transform_compound(self, compound):
        if self.cached_compounds and compound in self.cached_compounds:
            return self.cached_compounds[compound]
        else:
            tr_compound = self.compound_transform(compound)

            if self.use_compond_cache:
                self.cached_compounds[compound] = tr_compound
            return tr_compound
        
    def __len__(self):
        return len(self.compound_list)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={len(self.compound_list)})"
    
    def get_compound(self, idx):
        compound = self.compound_list[idx]
        if self.compound_transform:
            tr_compound = self.transform_compound(compound)
        else:
            tr_compound = compound
        return tr_compound
    
    def __getitem__(self, idx):
        tries = 0
        while tries < self.max_tries:
            try:
                compound = self.get_compound(idx)
                return compound
            except Exception as e:
                idx = random.randint(0, self.n_compounds - 1)
                tries += 1
                py_logger.warning(f"Could not get item {idx}. Try: {tries}/{self.max_tries}. Error: {e}")


class ImageDataset(Dataset):
    def __init__(
        self,
        image_list: List[List[str]],
        transform: Optional[Callable] = None,
        max_tries: int = 10,
    ):
        """Initializes the dataset.

        Args:
            image_list (List[List[str]]): A list of lists of image paths.
                Each element of the list is a 5 element list of image paths, once for each channels.
            transform (Optional[Callable], optional): A transform to apply to the
                image. Usually torchvision transforms. Defaults to None.
            max_tries (int, optional): The maximum number of tries to sample an image for a compound.
                Defaults to 10.
        """
        super().__init__()
        self.image_list = image_list
        self.transform = transform
        self.max_tries = max_tries
        
        self.n_images = len(self.image_list)
        
    def __len__(self):
        return self.n_images
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_images={self.n_images})"
    
    def get_image(self, idx):
        image_paths = self.image_list[idx]
        img_array = load_image_paths_to_array(image_paths)
        img_array = torch.from_numpy(img_array)
        if self.transform:
            img_array = self.transform(img_array)
        return img_array
    
    def __getitem__(self, idx):
        tries = 0
        while tries < self.max_tries:
            try:
                image = self.get_image(idx)
                return image
            except Exception as e:
                idx = random.randint(0, self.n_images - 1)
                tries += 1
                py_logger.warning(f"Could not get item {idx}. Try: {tries}/{self.max_tries}. Error: {e}")
                

class MoleculeImageDataset(MoleculeDataset, ImageDataset):
    def __init__(
        self,
        compound_dict: Dict[str, List[str]],
        load_df: pd.DataFrame,
        compound_str_type: Optional[Literal['smiles', 'inchi', 'selfies']] = None,
        compound_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        sampler: Optional[Callable] = None,
        channels: Sequence[str] = ("DNA", "AGP", "ER", "Mito", "RNA"),
        check_compound_transform: bool = False,
        use_compond_cache: bool = False,
        max_tries: int = 10,
        remove_bad: bool = True,
        verbose: bool = False
    ):
        self.load_df = load_df
        self.compound_dict = compound_dict
        self.sampler = sampler or random.choice
        self.channels = channels
        self.remove_bad = remove_bad
        
        self.compound_list = list(self.compound_dict.keys())
        self.setup_image_list()  # sets self.image_list and self.image_id_to_position
        
        # Sets the get_compound method
        super(MoleculeDataset, self).__init__(
            compound_list=self.compound_list,
            compound_str_type=compound_str_type,
            compound_transform=compound_transform,
            check_compound_transform=check_compound_transform,
            use_compond_cache=use_compond_cache,
            max_tries=max_tries,
            verbose=verbose
        )
        
        # Sets the get_image method
        super(ImageDataset, self).__init__(
            image_list=self.image_list,
            transform=transform,
            max_tries=max_tries
        )
    
    def setup_image_list(self):
        cols = [f"FileName_Orig{channel}" for channel in self.channels]
        # Use the values of self.compound_dict to get the image ids
        index = [image_id for image_ids in self.compound_dict.values() for image_id in image_ids]
        index = list(set(index))  # Remove duplicates
        
        # This allows to keep only the images that are associated with a compound
        
        self.image_list = self.load_df.loc[index, cols].values  # n_images x 5
        self.image_id_to_position = {image_id: i for i, image_id in enumerate(self.load_df.index.tolist())}
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_compounds={self.n_compounds}, n_images={self.n_images})"
    
    def __len__(self):
        return self.n_compounds
    
    def get_sample(self, compound_id):
        """Returns an image id."""
        return self.sampler(self.compound_dict[compound_id])
    
    def remove_compound(self, idx):
        """Removes a compound from the compound list."""
        compound_id = self.compound_list[idx]               # An inchi or smiles string
                
        del self.compound_list[idx]                         # Remove the compound from the compound_list
        del self.compound_dict[compound_id]                 # Remove the compound from the compound_dict
        
        self.n_compounds -= 1                               # Decrease the number of compounds
        
    def remove_image(self, image_id, compound_id):
        """Removes an image from the image list."""
        del self.image_id_to_position[image_id]             # Remove the image from the image_id_to_position dict
        
        # Remove the image from the compound_dict
        self.compound_dict[compound_id] = [i for i in self.compound_dict[compound_id] if i != image_id]
        
        self.n_images -= 1                                  # Decrease the number of images
    
    def __getitem__(self, idx):
        tries = 0
        
        while tries < self.max_tries:
            try:
                compound_id = self.compound_list[idx]   # An inchi or smiles string
                compound = self.get_compound(idx)       # The chosen representation of the compound (e.g. a graph)
            except Exception as e:
                if self.remove_bad:
                    self.remove_compound(idx)
                idx = random.randint(0, self.n_compounds - 1)
                tries += 1
                py_logger.warning(f"Could not get compound {idx}. Try: {tries}/{self.max_tries}. Error: {e}")
                continue
            
            image_tries = 0
            image_tried = []
            while image_tries < self.max_tries:
                try:
                    image_id = self.get_sample(compound_id)             # An image id (index in the load_df)
                    image_tried.append(image_id)
                    image_pos = self.image_id_to_position[image_id]     # The id in the image_list
                    image = self.get_image(image_pos)                   # The image (a tensor or numpy array)
                    return {"image": image, "compound": compound}
                except UnidentifiedImageError:
                    if self.remove_bad:
                        self.remove_image(image_id, compound_id)
                    image_tries += 1
                    py_logger.warning(f"Could not load image {image_id}. Try: {tries}/{self.max_tries}")
        
            # If we get here, we could not find an image for the compound, so we try another compound
            if self.remove_bad:
                self.remove_compound(idx)
            idx = random.randint(0, self.n_compounds - 1)
            tries += 1
            py_logger.warning(f"Could not get compound {idx}. Try: {tries}/{self.max_tries}. Error: {e}")
