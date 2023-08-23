"""Defines a callback that allows to explore the loss when it becomes NaN."""

from glob import glob
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback

from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class NaNLossCallback(Callback):
    def __init__(
        self,
        max_files: int = 10,
    ):
        self.max_files = max_files
        super().__init__()

    def setup(self, trainer, pl_module, stage=None):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        nan = torch.tensor(float("nan"))

        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif outputs is None:
            loss = nan
        elif isinstance(outputs, dict):
            loss = outputs.get("loss", nan)

        if not torch.isfinite(loss):
            py_logger.info(f"Loss of batch {batch_idx} is {loss}. Returning None.")
            py_logger.info(f"Batch: {batch}")
            out_file = (
                Path(trainer.log_dir) / f"{trainer.state.status}_epoch_{trainer.current_epoch}_batch_{batch_idx}.pt"
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            num_files = len(glob(str(out_file.parent / "*.pt")))

            if num_files < self.max_files:
                torch.save(batch, out_file)
