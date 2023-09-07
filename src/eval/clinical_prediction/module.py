"""LightningModule for HINT evalulation.

Inspired by https://github.com/futianfan/clinical-trial-outcome-
prediction/blob/main/HINT/molecule_encode.py#L92
"""


import copy
from typing import Any, Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.modules.collate_fn import default_collate
from src.modules.lr_schedulers.warmup_wrapper import WarmUpWrapper
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class HintClinicalModule(LightningModule):
    prefix: str = "hint"
    phase: Optional[Literal["I", "II", "III"]] = None

    default_criterion = nn.BCEWithLogitsLoss

    additional_metrics = [
        BinaryAUROC(),
        BinaryAccuracy(),
        BinaryRecall(),
        BinaryPrecision(),
        BinaryF1Score(),
    ]
    plot_metrics = [
        BinaryConfusionMatrix(),
    ]

    def __init__(
        self,
        cross_modal_module: Optional[LightningModule] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        criterion: Optional[torch.nn.Module] = None,
        molecule_encoder: Optional[nn.Module] = None,
        molecule_encoder_attribute_name: str = "molecule_encoder",
        compound_transform: Optional[Callable] = None,
        example_input: Optional[torch.Tensor] = None,
        example_input_path: Optional[str] = None,
        lr: float = 1e-3,
        split_lr_in_groups: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "cross_modal_module",
                "optimizer",
                "criterion",
                "molecule_encoder",
                "molecule_encoder_attribute_name",
            ],
        )

        # encoder
        if not (molecule_encoder or (cross_modal_module and molecule_encoder_attribute_name)):
            raise ValueError("Either molecule_encoder or cross_modal_module with attribute name must be provided.")

        if molecule_encoder is not None:
            self.molecule_encoder = copy.deepcopy(molecule_encoder)
            self.model_name = molecule_encoder.__class__.__name__
        else:
            self.molecule_encoder = copy.deepcopy(getattr(cross_modal_module, molecule_encoder_attribute_name))
            self.model_name = self.molecule_encoder.__class__.__name__

        # compound transform
        self.compound_transform = compound_transform
        if self.compound_transform is not None:
            self.compound_transform.compound_str_type = "smiles"

        # head
        self.embedding_dim = self.molecule_encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
        )

        # loss function
        if criterion is None:
            self.criterion = self.default_criterion()
        else:
            self.criterion = criterion

        # training
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.split_lr_in_groups = split_lr_in_groups

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
        self.log_prefix = f"{self.prefix}/phase_{self.phase}"

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

    def get_batched_graphs(self, smiles_lst_lst):
        ids = []
        graphs = []

        for i, smiles_lst in enumerate(smiles_lst_lst):
            for smiles in smiles_lst:
                if self.compound_transform is not None:
                    smiles = self.compound_transform(smiles)

                if smiles is not None:
                    graphs.append(smiles)
                    ids.append(i)

        batched_graphs = default_collate(graphs).to(self.device)

        return batched_graphs, ids

    def forward_smiles_lst_lst(self, smiles_lst_lst):
        batched_graphs, ids = self.get_batched_graphs(smiles_lst_lst)

        b_emb = self.molecule_encoder(batched_graphs)
        ids = torch.IntTensor(ids).to(self.device)

        # Average the embeddings wrt the ids
        bincount = torch.bincount(ids, minlength=len(ids))
        numerator = torch.zeros_like(b_emb)
        numerator = numerator.index_add(0, ids, b_emb)
        non_empty_ids = bincount != 0
        div = bincount.float()[non_empty_ids]

        mean = torch.div(numerator[non_empty_ids].t(), div).t()

        return mean

    def forward(self, compound, **kwargs):
        return self.molecule_encoder(compound)

    def model_step(self, batch: Any, stage: str = "train", on_step_loss=True):
        smiles_list = batch["smiles_list"]
        targets = batch["label"].float().view(-1, 1)

        compound_embeddings = self.forward_smiles_lst_lst(smiles_list)
        logits = self.head(compound_embeddings)

        loss = self.criterion(logits, targets)
        preds = F.sigmoid(logits)

        # update metrics
        self.loss_dict[stage](loss)
        self.plot_metrics_dict[stage](preds, targets)
        self.other_metrics_dict[stage](preds, targets)

        # log metrics
        self.log(
            f"{self.log_prefix}/{stage}/loss", self.loss_dict[stage], on_step=on_step_loss, on_epoch=True, prog_bar=True
        )
        self.log_dict(self.other_metrics_dict[stage](preds, targets), on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def training_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, stage="train", on_step_loss=True)

    def validation_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, stage="val", on_step_loss=False)

    def test_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, stage="test", on_step_loss=False)

    def on_train_start(self):
        self.val_loss.reset()
        super().on_train_start()

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def split_groups(self):
        params_groups = [
            {
                "params": list(self.head.parameters()),
                "name": f"{self.log_prefix}_head",
            },
            {
                "params": list(self.molecule_encoder.parameters()),
                "name": f"{self.log_prefix}_molecule_encoder",
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
            optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

        # optimizer = self.optimizer(params=filter(lambda p: p.requires_grad, self.parameters()))
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)

            lr_scheduler_dict = {
                "scheduler": scheduler,
                "monitor": f"{self.log_prefix}/val/loss",
                "interval": "epoch",
                "frequency": 1,
                "name": f"{self.log_prefix}/lr",
            }

            if isinstance(scheduler, WarmUpWrapper) and isinstance(scheduler.wrapped_scheduler, ReduceLROnPlateau):
                lr_scheduler_dict["reduce_on_plateau"] = True

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_dict,
            }
        return {"optimizer": optimizer}


class HintClinicalModulePhaseI(HintClinicalModule):
    phase = "I"


class HintClinicalModulePhaseII(HintClinicalModule):
    phase = "II"


class HintClinicalModulePhaseIII(HintClinicalModule):
    phase = "III"
