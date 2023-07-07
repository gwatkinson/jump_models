import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


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
