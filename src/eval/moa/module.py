"""LightningModule for Jump MOA datasets evalulation."""
# flake8: noqa

from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class JumpMOAImageModule(LightningModule):
    num_classes = 26
    prefix = "jump_moa"
    dataset_name = "image"
    default_criterion = nn.CrossEntropyLoss
    additional_metrics = {
        "MulticlassAUROC_top_1": MulticlassAUROC(num_classes=26, average="weighted"),
        "MulticlassAccuracy_top_1": MulticlassAccuracy(num_classes=26, average="weighted", top_k=1),
        "MulticlassAccuracy_top_3": MulticlassAccuracy(num_classes=26, average="weighted", top_k=3),
        "MulticlassAccuracy_top_5": MulticlassAccuracy(num_classes=26, average="weighted", top_k=5),
        "MulticlassAccuracy_top_10": MulticlassAccuracy(num_classes=26, average="weighted", top_k=10),
        "MulticlassF1Score_top_1": MulticlassF1Score(num_classes=26, average="weighted", top_k=1),
        "MulticlassF1Score_top_5": MulticlassF1Score(num_classes=26, average="weighted", top_k=5),
    }
    plot_metrics = {
        "MulticlassConfusionMatrix": MulticlassConfusionMatrix(num_classes=26, normalize=None),
        "MulticlassConfusionMatrix_normalized": MulticlassConfusionMatrix(num_classes=26, normalize="true"),
    }

    def __init__(
        self,
        cross_modal_module: Optional[LightningModule],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Optional[torch.nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        image_encoder_attribute_name: str = "image_encoder",
        example_input: Optional[torch.Tensor] = None,
        example_input_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["cross_modal_module", "criterion", "image_encoder"])

        # encoder
        if image_encoder is not None:
            self.image_encoder = image_encoder
        else:
            self.image_encoder = getattr(cross_modal_module, image_encoder_attribute_name)
        self.embedding_dim = self.image_encoder.out_dim
        self.head = nn.Linear(self.embedding_dim, self.num_classes)

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
        other_metric_collection = MetricCollection(self.additional_metrics)
        self.train_other_metrics = other_metric_collection.clone(prefix=f"{self.log_prefix}/train/")
        self.val_other_metrics = other_metric_collection.clone(prefix=f"{self.log_prefix}/val/")
        self.test_other_metrics = other_metric_collection.clone(prefix=f"{self.log_prefix}/test/")

        plot_metric_collection = MetricCollection(self.plot_metrics)
        self.train_plot_metrics = plot_metric_collection.clone(prefix=f"{self.log_prefix}/train/")
        self.val_plot_metrics = plot_metric_collection.clone(prefix=f"{self.log_prefix}/val/")
        self.test_plot_metrics = plot_metric_collection.clone(prefix=f"{self.log_prefix}/test/")

        self.loss_dict = {
            "train": self.train_loss,
            "val": self.val_loss,
            "test": self.test_loss,
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

    def extract(self, image):
        return self.image_encoder(image)

    def forward(self, image, label=None, compound=None):
        emb = self.extract(image)
        return self.head(emb)

    def model_step(self, batch: Any, stage: str = "train", on_step_loss=True):
        targets = batch["label"]
        batch_size = targets.shape[0]
        logits = self.forward(**batch)
        loss = self.criterion(logits, targets)

        # update metrics
        self.loss_dict[stage](loss)
        self.plot_metrics_dict[stage].update(logits, targets)
        other_metrics = self.other_metrics_dict[stage](logits, targets)

        # log metrics
        self.log(
            f"{self.log_prefix}/{stage}/loss",
            self.loss_dict[stage],
            on_step=on_step_loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log_dict(other_metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

        return loss, logits, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, _preds, _targets = self.model_step(batch, stage="train", on_step_loss=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        _loss, _preds, _targets = self.model_step(batch, stage="val", on_step_loss=False)

    def on_validation_epoch_end(self):
        pass

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


class JumpMOAImageGraphModule(JumpMOAImageModule):
    dataset_name = "image_graph"

    def __init__(
        self,
        cross_modal_module: Optional[LightningModule],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Optional[torch.nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        molecule_encoder: Optional[nn.Module] = None,
        image_encoder_attribute_name: str = "image_encoder",
        molecule_encoder_attribute_name: str = "molecule_encoder",
        example_input: Optional[torch.Tensor] = None,
        example_input_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            cross_modal_module=cross_modal_module,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            image_encoder=image_encoder,
            image_encoder_attribute_name=image_encoder_attribute_name,
            example_input=example_input,
            example_input_path=example_input_path,
        )
        self.save_hyperparameters(
            logger=False, ignore=["cross_modal_module", "criterion", "image_encoder", "molecule_encoder"]
        )

        # encoders and head
        if molecule_encoder is not None:
            self.molecule_encoder = molecule_encoder
        else:
            self.molecule_encoder = getattr(cross_modal_module, molecule_encoder_attribute_name)
        self.molecule_embedding_dim = self.molecule_encoder.out_dim
        self.image_embedding_dim = self.image_encoder.out_dim
        self.embedding_dim = self.molecule_embedding_dim + self.image_embedding_dim
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

    def extract(self, image, compound):
        image_emb = self.image_encoder(image)
        compound_emb = self.molecule_encoder(compound)
        return torch.cat([image_emb, compound_emb], dim=1)

    def forward(self, image, compound, label=None):
        emb = self.extract(image, compound)
        return self.head(emb)
