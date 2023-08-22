"""Defines a callback that allows to explore the loss to check it is
correct."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch.loggers as pl_loggers
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

import wandb
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


class LossCheckCallback(Callback):
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
        }

    def on_batch_end(self, phase: str, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, dict):
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

    @staticmethod
    def get_loss_from_list_of_dict(losses: List[Dict[str, Any]]):
        return pd.DataFrame(losses)

    @staticmethod
    def get_loss_by_epoch(losses: List[Dict[str, Any]]):
        return pd.DataFrame(losses).groupby("epoch")["loss"].sum().reset_index()

    def on_fit_end(self, trainer, pl_module):
        train_step_loss = self.get_loss_from_list_of_dict(self.losses["train"])
        val_step_loss = self.get_loss_from_list_of_dict(self.losses["val"])

        train_epoch_loss = self.get_loss_by_epoch(self.losses["train"])
        val_epoch_loss = self.get_loss_by_epoch(self.losses["val"])

        fig, ax = plt.subplots()
        ax.plot(train_epoch_loss["epoch"], train_epoch_loss["loss"], label="train")
        ax.plot(val_epoch_loss["epoch"], val_epoch_loss["loss"], label="val")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss by epoch")

        if self.image_dir is not None:
            py_logger.info(f"Saving loss plot to {self.image_dir}")
            image_dir = Path(self.image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)

            train_epoch_loss.to_csv(image_dir / "train_epoch_loss.csv")
            val_epoch_loss.to_csv(image_dir / "val_epoch_loss.csv")

            train_step_loss.to_csv(image_dir / "train_step_loss.csv")
            val_step_loss.to_csv(image_dir / "val_step_loss.csv")

            fig.savefig(str(image_dir / "loss_by_epoch.png"))

        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                wandb_logger = logger.experiment
                break

        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "callback/loss_by_epoch": wandb.Image(fig),
                    "callback/train_epoch_loss": wandb.Table(data=train_epoch_loss),
                    "callback/val_epoch_loss": wandb.Table(data=val_epoch_loss),
                    "callback/train_step_loss": wandb.Table(data=train_step_loss),
                    "callback/val_step_loss": wandb.Table(data=val_step_loss),
                }
            )

        plt.close(fig)
