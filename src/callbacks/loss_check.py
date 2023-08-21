"""Defines a callback that allows to explore the loss to check it is
correct."""

import os.path as osp
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class NaNLossCallback(Callback):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        image_dir: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.image_dir = image_dir

    def setup(self, trainer, pl_module, stage=None):
        pass

    def on_fit_start(self, trainer, pl_module):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
        }

    def on_batch_end(self, phase: str, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, Dict[str, torch.Tensor]):
            loss = outputs["loss"]

        if "image" in batch:
            n = len(batch["image"])
        elif "label" in batch:
            n = len(batch["label"])
        elif self.batch_size is not None:
            n = self.batch_size
        else:
            raise ValueError("Could not find batch size.")

        self.losses[phase].append(
            {
                "loss": loss,
                "n": n,
                "batch_idx": batch_idx,
                "epoch": trainer.current_epoch,
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_batch_end("train", trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_batch_end("val", trainer, pl_module, outputs, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_batch_end("test", trainer, pl_module, outputs, batch, batch_idx)

    @staticmethod
    def get_loss_from_list_of_dict(losses: List[Dict[str, Any]]):
        return [loss["loss"] for loss in losses]

    @staticmethod
    def get_loss_by_epoch(losses: List[Dict[str, Any]]):
        return pd.DataFrame(losses).groupby("epoch")["loss"].sum().reset_index()

    def on_fit_end(self, trainer, pl_module):
        train_step_loss = self.get_loss_from_list_of_dict(self.losses["train"])
        val_step_loss = self.get_loss_from_list_of_dict(self.losses["val"])
        test_step_loss = self.get_loss_from_list_of_dict(self.losses["test"])

        py_logger.info(f"Train step loss: {train_step_loss}")
        py_logger.info(f"Val step loss: {val_step_loss}")
        py_logger.info(f"Test step loss: {test_step_loss}")

        train_epoch_loss = self.get_loss_by_epoch(self.losses["train"])
        val_epoch_loss = self.get_loss_by_epoch(self.losses["val"])
        test_epoch_loss = self.get_loss_by_epoch(self.losses["test"])

        py_logger.info(f"Train epoch loss: {train_epoch_loss}")
        py_logger.info(f"Val epoch loss: {val_epoch_loss}")
        py_logger.info(f"Test epoch loss: {test_epoch_loss}")

        fig, ax = plt.subplots()
        ax.plot(train_epoch_loss["epoch"], train_epoch_loss["loss"], label="train")
        ax.plot(val_epoch_loss["epoch"], val_epoch_loss["loss"], label="val")
        ax.plot(test_epoch_loss["epoch"], test_epoch_loss["loss"], label="test")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss by epoch")

        if self.image_dir is not None:
            fig.savefig(osp.join(self.image_dir, "loss_by_epoch.png"))
        plt.close(fig)
