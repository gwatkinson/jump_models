"""Defines a callback that allows to explore the loss when it becomes NaN."""

from typing import Dict

import torch
from lightning.pytorch.callbacks import Callback

from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class NaNLossCallback(Callback):
    def __init__(
        self,
    ):
        super().__init__()

    def setup(self, trainer, pl_module, stage=None):
        pass

    def on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, Dict[str, torch.Tensor]):
            loss = outputs["loss"]

        if not torch.isfinite(loss):
            py_logger.info(f"Loss of batch {batch_idx} is {loss}. Returning None.")
            py_logger.info(f"Batch: {batch}")
