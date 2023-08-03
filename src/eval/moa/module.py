"""LightningModule for OGB datasets evalulation."""
# flake8: noqa

from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class MOAClassificationModule(LightningModule):
    """Module for evaluating a model on the MOA tasks from the images."""

    prefix = "moa"
    dataset_name = None
    out_dim = 1
    default_criterion = nn.BCEWithLogitsLoss
    additional_metrics = [
        BinaryAUROC,
        BinaryAccuracy,
        BinaryRecall,
        BinaryPrecision,
        BinaryF1Score,
    ]
    plot_metrics = [
        BinaryConfusionMatrix,
    ]

    def __init__(
        self,
        cross_modal_module: LightningModule,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Optional[torch.nn.Module] = None,
        molecule_encoder_attribute_name: str = "molecule_encoder",
        example_input: Optional[torch.Tensor] = None,
        example_input_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["cross_modal_module", "criterion"])

        # encoder
        # self.cross_modal_module = cross_modal_module
        self.molecule_encoder = getattr(cross_modal_module, molecule_encoder_attribute_name)
        self.embedding_dim = getattr(self.molecule_encoder, "out_dim", None)
        self.head = nn.Linear(self.embedding_dim, self.out_dim)

        # model
        self.model = nn.Sequential(
            self.molecule_encoder,
            self.head,
        )

        # loss function
        if criterion is None:
            self.criterion = self.default_criterion()
        else:
            self.criterion = criterion

        # training
        self.optimizer = optimizer
        self.scheduler = scheduler

        # example input
        if example_input is not None:
            self.example_input_array = example_input
        elif example_input_path is not None:
            self.example_input_array = torch.load(example_input_path)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # metric objects for calculating and averaging accuracy across batches
        self.log_prefix = f"{self.prefix}/{self.dataset_name}"

        self.train_other_metrics = MetricCollection(
            [metric() for metric in self.additional_metrics], prefix=f"{self.log_prefix}/train/"
        )
        self.val_other_metrics = self.train_other_metrics.clone(prefix=f"{self.log_prefix}/val/")
        self.test_other_metrics = self.train_other_metrics.clone(prefix=f"{self.log_prefix}/test/")

        self.train_plot_metrics = MetricCollection(
            [metric() for metric in self.plot_metrics], prefix=f"{self.log_prefix}/train/"
        )
        self.val_plot_metrics = self.train_plot_metrics.clone(prefix=f"{self.log_prefix}/val/")
        self.test_plot_metrics = self.train_plot_metrics.clone(prefix=f"{self.log_prefix}/test/")

        self.loss_dict = {
            "train": self.train_loss,
            "val": self.val_loss,
            "test": self.test_loss,
        }
        self.metric_dict = {
            "train": self.train_metric,
            "val": self.val_metric,
            "test": self.test_metric,
        }
        self.other_metrics_dict = {
            "train": self.train_other_metrics,
            "val": self.val_other_metrics,
            "test": self.test_other_metrics,
        }
        self.plot_metrics_dict = {
            "train": self.train_plot_metrics,
            "val": self.val_plot_metrics,
            "test": self.test_plot_metrics,
        }

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        super().on_train_start()

    def extract(self, compound):
        return self.molecule_encoder(compound)

    def forward(self, compound, label=None):
        return self.model(compound)

    def model_step(self, batch: Any, stage: str = "train", on_step_loss=True):
        compound = batch["compound"]
        targets = batch["label"]

        logits = self.model(compound)

        loss = self.criterion(logits, targets)
        # preds = F.sigmoid(logits)  # is in the torchmetrics implementation ??

        # update metrics
        self.loss_dict[stage].update(loss)
        self.metric_dict[stage].update(logits, targets)
        self.plot_metrics_dict[stage].update(logits, targets)
        other_metrics = self.other_metrics_dict[stage](logits, targets)

        # log metrics
        self.log("{stage}/loss", self.loss_dict[stage], on_step=on_step_loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/{self.metric_name}", self.metric_dict[stage], on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(other_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss, logits, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, _preds, _targets = self.model_step(batch, stage="train", on_step_loss=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        _loss, _preds, _targets = self.model_step(batch, stage="val", on_step_loss=False)

    def on_validation_epoch_end(self):
        metric = self.val_metric.compute()
        self.val_metric_best(metric)
        self.log(f"val/{self.metric_name}_best", self.val_metric_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch, stage="test", on_step_loss=False)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=filter(lambda p: p.requires_grad, self.parameters()))
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}