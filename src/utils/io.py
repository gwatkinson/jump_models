import json
from io import BytesIO
from typing import Any, Dict, List, Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from PIL import Image


def download_and_extract_zip(url: str, path: str) -> None:
    """Download a zip file from a url and extract it to a path."""
    resp = requests.get(url)
    zipfile = ZipFile(BytesIO(resp.content))
    zipfile.extractall(path)


def load_load_df_from_parquet(path: str, job_dtypes: Optional[Dict[str, Any]] = None, **kwargs) -> pd.DataFrame:
    """Load the load_data df from a csv file using only the columns specified
    in the job_dtypes dict."""

    job_dtypes = job_dtypes or {
        "Metadata_Source": object,
        "Metadata_Batch": object,
        "Metadata_Plate": object,
        "Metadata_Well": object,
        "Metadata_Site": np.int64,
        "FileName_OrigDNA": object,
        "FileName_OrigAGP": object,
        "FileName_OrigER": object,
        "FileName_OrigMito": object,
        "FileName_OrigRNA": object,
    }

    load_df = pd.read_parquet(path, **kwargs)
    load_df = load_df.astype(job_dtypes)

    return load_df


def load_metadata_df_from_csv(path: str, job_dtypes: Optional[Dict[str, Any]] = None, **kwargs) -> pd.DataFrame:
    """Load the complete_metadata df from a csv file using only the columns
    specified in the job_dtypes dict."""

    job_dtypes = job_dtypes or {
        "Metadata_Source": object,
        "Metadata_Batch": object,
        "Metadata_Plate": object,
        "Metadata_Well": object,
        "Metadata_JCP2022": object,
        "Metadata_InChI": object,
        "Metadata_PlateType": object,
    }
    cols_to_keep = list(job_dtypes.keys())

    metadata_df = pd.read_csv(path, dtype=job_dtypes, usecols=cols_to_keep, **kwargs)

    return metadata_df


def load_dict_from_json(path: str) -> Dict[str, Any]:
    """Load the compound_dict from a json file."""

    with open(path) as f:
        compound_dict = json.load(f)

    return compound_dict


def load_image_paths_to_array(image_paths: List[str]):
    """Load a list of image paths into a numpy array."""
    images = []
    for image_path in image_paths:
        images.append(np.array(Image.open(image_path)))
    return np.stack(images)
