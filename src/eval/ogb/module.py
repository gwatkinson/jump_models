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
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

RMSE = partial(MeanSquaredError, squared=False)


class OGBClassificationModule(LightningModule):
    """Module for evaluating a model on the OGB classification tasks."""

    metric_name = "AUC"
    task_metric = BinaryAUROC
    best_metric = MaxMetric
    out_dim = 1
    additional_metrics = [
        BinaryAccuracy,
        BinaryRecall,
        BinaryPrecision,
        BinaryF1Score,
    ]
    plot_metrics = [
        BinaryConfusionMatrix,
    ]
    default_criterion = nn.BCEWithLogitsLoss

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
        self.train_metric = self.task_metric()
        self.val_metric = self.task_metric()
        self.test_metric = self.task_metric()
        self.val_metric_best = self.best_metric()

        self.train_other_metrics = MetricCollection([metric() for metric in self.additional_metrics], prefix="train/")
        self.val_other_metrics = MetricCollection([metric() for metric in self.additional_metrics], prefix="val/")
        self.test_other_metrics = MetricCollection([metric() for metric in self.additional_metrics], prefix="test/")

        self.train_plot_metrics = MetricCollection([metric() for metric in self.plot_metrics], prefix="train/")
        self.val_plot_metrics = MetricCollection([metric() for metric in self.plot_metrics], prefix="val/")
        self.test_plot_metrics = MetricCollection([metric() for metric in self.plot_metrics], prefix="test/")

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        super().on_train_start()

    def extract(self, compound):
        return self.molecule_encoder(compound)

    def forward(self, compound, label=None):
        return self.model(compound)

    def model_step(self, batch: Any, phase: str = "train", on_step_loss=True):
        compound = batch["compound"]
        targets = batch["label"]

        logits = self.model(compound)

        loss = self.criterion(logits, targets)
        # preds = F.sigmoid(logits)  # is in the torchmetrics implementation ??

        return loss, logits, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update metrics
        self.train_loss(loss)
        self.train_metric(preds, targets)
        self.train_plot_metrics(preds, targets)
        other_metrics = self.train_other_metrics(preds, targets)

        # log metrics
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/{self.metric_name}", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(other_metrics, on_step=False, on_epoch=True, prog_bar=False)

        # return loss or backpropagation will fail
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update metrics
        self.val_loss(loss)
        self.val_metric(preds, targets)
        self.val_plot_metrics(preds, targets)
        other_metrics = self.train_other_metrics(preds, targets)

        # log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/{self.metric_name}", self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(other_metrics, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self):
        metric = self.val_metric.compute()  # get current val metric
        self.val_metric_best(metric)  # update best so far val metric
        # log `val_metric_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(f"val/{self.metric_name}_best", self.val_metric_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update metrics
        self.test_loss(loss)
        self.test_metric(preds, targets)
        self.test_plot_metrics(preds, targets)
        other_metrics = self.train_other_metrics(preds, targets)

        # log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test/{self.metric_name}", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(other_metrics, on_step=False, on_epoch=True, prog_bar=False)

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
        # optimizer = self.optimizer(params=self.parameters())
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


class OGBRegressionModule(OGBClassificationModule):
    """Module for evaluating a model on the OGB classification tasks with RMSE
    as the main metric."""

    metric_name = "rmse"
    task_metric = RMSE
    best_metric = MinMetric
    out_dim = 1
    additional_metrics = [
        MeanAbsoluteError,
        R2Score,
    ]
    plot_metrics = []

    default_criterion = nn.MSELoss


class BBBPModule(OGBClassificationModule):
    dataset_name = "bbbp"


class HIVModule(OGBClassificationModule):
    dataset_name = "hiv"


class Tox21Module(OGBClassificationModule):
    dataset_name = "tox21"


class EsolModule(OGBRegressionModule):
    dataset_name = "esol"


class LipoModule(OGBRegressionModule):
    dataset_name = "lipophilicity"


class ToxCastModule(OGBClassificationModule):  # TODO: Not a simple 1 label classification task
    dataset_name = "toxcast"
