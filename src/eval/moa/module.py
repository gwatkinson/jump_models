"""LightningModule for Jump MOA datasets evalulation."""

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from src.modules.lr_schedulers.warmup_wrapper import WarmUpWrapper
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class JumpMOAImageModule(LightningModule):
    prefix = "jump_moa"
    dataset_name = "image"
    default_criterion = nn.CrossEntropyLoss

    def __init__(
        self,
        cross_modal_module: Optional[LightningModule] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        criterion: Optional[torch.nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        image_encoder_attribute_name: str = "image_encoder",
        example_input: Optional[torch.Tensor] = None,
        example_input_path: Optional[str] = None,
        lr: float = 1e-3,
        num_classes: int = 26,
        split_lr_in_groups: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.additional_metrics = {
            "AUROC": MulticlassAUROC(num_classes=self.num_classes, average="weighted"),
            "Accuracy_top_1": MulticlassAccuracy(num_classes=self.num_classes, average="weighted", top_k=1),
            "Accuracy_top_3": MulticlassAccuracy(num_classes=self.num_classes, average="weighted", top_k=3),
            "Accuracy_top_5": MulticlassAccuracy(num_classes=self.num_classes, average="weighted", top_k=5),
            "F1Score_top_1": MulticlassF1Score(num_classes=self.num_classes, average="weighted", top_k=1),
            "F1Score_top_5": MulticlassF1Score(num_classes=self.num_classes, average="weighted", top_k=5),
        }
        self.plot_metrics = {
            "ConfusionMatrix": MulticlassConfusionMatrix(num_classes=self.num_classes, normalize=None),
        }

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["cross_modal_module", "optimizer", "criterion", "image_encoder", "image_encoder_attribute_name"],
        )

        # encoder
        if not (image_encoder or (cross_modal_module and image_encoder_attribute_name)):
            raise ValueError("Either image_encoder or cross_modal_module with attribute name must be provided.")

        if image_encoder is not None:
            self.image_encoder = copy.deepcopy(image_encoder)
            self.model_name = image_encoder.__class__.__name__
        else:
            self.image_encoder = copy.deepcopy(getattr(cross_modal_module, image_encoder_attribute_name))
            self.model_name = self.image_encoder.__class__.__name__

        self.embedding_dim = self.image_encoder.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.2),
            nn.Linear(self.embedding_dim, self.num_classes),
        )
        self.lr = lr
        self.split_lr_in_groups = split_lr_in_groups

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

    def __repr__(self):
        return f"""{self.__class__.__name__}({self.model_name}({self.embedding_dim}), num_classes={self.num_classes})"""

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        super().on_train_start()

    def extract(self, image):
        return self.image_encoder(image)

    def forward(self, image, **kwargs):
        emb = self.extract(image)
        return self.head(emb)

    def model_step(self, batch: Any, stage: str = "train", on_step_loss=True):
        targets = batch["label"]
        batch_size = targets.shape[0]
        logits = self.forward(**batch)
        loss = self.criterion(logits, targets)
        preds = F.softmax(logits, dim=1)

        # update metrics
        self.loss_dict[stage](loss)
        self.plot_metrics_dict[stage](preds, targets)
        self.other_metrics_dict[stage](preds, targets)

        # log metrics
        self.log(
            f"{self.log_prefix}/{stage}/loss",
            self.loss_dict[stage],
            on_step=on_step_loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log_dict(
            self.other_metrics_dict[stage], on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size
        )

        # if not torch.isfinite(loss):  # TODO: Make this device agnostic
        #     loss = None

        return {"loss": loss}

    def training_step(self, batch: Any, batch_idx: int):
        out = self.model_step(batch, stage="train", on_step_loss=True)
        return out["loss"]

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        out = self.model_step(batch, stage="val", on_step_loss=False)
        return out["loss"]

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        out = self.model_step(batch, stage="test", on_step_loss=False)
        return out["loss"]

    def on_test_epoch_end(self):
        pass

    def split_groups(self):
        params_groups = [
            {
                "params": list(self.head.parameters()),
                "name": "moa_image_head",
            },
            {
                "params": list(self.image_encoder.parameters()),
                "name": "moa_image_encoder",
            },
        ]
        filtered_params_groups = [
            {
                "params": list(filter(lambda p: p.requires_grad, group["params"])),
                "name": group["name"],
            }
            for group in params_groups
        ]

        params_len = {group["name"]: len(group["params"]) for group in params_groups}
        group_lens = {group["name"]: len(group["params"]) for group in filtered_params_groups}

        group_to_keep = [group["name"] for group in filtered_params_groups if group_lens[group["name"]] > 0]

        logger.info(f"Number of params in each groups: {params_len}")
        logger.info(f"Number of require grad params in each groups: {group_lens}")

        optimizer = self.optimizer(
            [group for group in params_groups if group["name"] in group_to_keep],
            lr=self.lr,
        )

        return optimizer

    def configure_optimizers(self):
        if self.split_lr_in_groups:
            optimizer = self.split_groups()
        else:
            optimizer = self.optimizer(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, name="learning_rate"
            )

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)

            lr_scheduler_dict = {
                "scheduler": scheduler,
                "monitor": "jump_moa/image/val/loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "jump_moa/image/lr",
            }

            if isinstance(scheduler, WarmUpWrapper) and isinstance(scheduler.wrapped_scheduler, ReduceLROnPlateau):
                lr_scheduler_dict["reduce_on_plateau"] = True

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_dict,
            }

        return {"optimizer": optimizer}
