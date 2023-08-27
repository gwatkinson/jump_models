"""Molecule encoder using the models from
https://github.com/terraytherapeutics/COATI/tree/main."""

import dgllife
import torch.nn as nn

from src.coati.models.io import load_e3gnn_smiles_clip_e2e
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


# TODO
