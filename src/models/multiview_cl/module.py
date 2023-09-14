from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric

from src.modules.lr_schedulers.warmup_wrapper import WarmUpWrapper
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class MutliviewModule(LightningModule):
    def __init__(
        self,
        image_encoder: torch.nn.Module,
        molecule_encoder: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        embedding_dim: int,
        example_input: Optional[torch.Tensor] = None,
        example_input_path: Optional[str] = None,
        monitor: str = "val/loss",
        interval: str = "epoch",
        frequency: int = 1,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["image_encoder", "molecule_encoder", "criterion"])

        # encoders
        self.image_encoder = image_encoder
        self.molecule_encoder = molecule_encoder
        self.criterion = criterion

        # projection heads
        self.image_dim = self.image_encoder.out_dim
        self.molecule_dim = self.molecule_encoder.out_dim

        self.image_projection_head = nn.Sequential(
            nn.ReLU(), nn.Dropout(kwargs.get("dropout", 0.1)), nn.Linear(self.image_dim, embedding_dim)
        )
        self.molecule_projection_head = nn.Sequential(
            nn.ReLU(), nn.Dropout(kwargs.get("dropout", 0.1)), nn.Linear(self.molecule_dim, embedding_dim)
        )

        # embedding dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lr = lr

        # training
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor
        self.interval = interval
        self.frequency = frequency

        # for averaging loss across batches
        self.train_loss = MeanMetric(nan_strategy="ignore")
        self.val_loss = MeanMetric(nan_strategy="ignore")
        self.test_loss = MeanMetric(nan_strategy="ignore")

        self.loss_dict = {
            "train": self.train_loss,
            "val": self.val_loss,
            "test": self.test_loss,
        }

        if example_input is not None:
            self.example_input_array = example_input
        elif example_input_path is not None:
            logger.debug(f"Loading example input from: {example_input_path}")
            self.example_input_array = torch.load(example_input_path)

    def forward(self, image, compound, **kwargs):
        batch_size, n_views, *image_shape = image.shape
        image = image.view(-1, *image_shape)
        image_emb = self.image_encoder(image)  # (BatchSize * Views) x ImagesDim

        image_emb = self.image_projection_head(image_emb)  # (B * V) x EmbeddingDim
        image_emb = F.normalize(image_emb, dim=-1)
        image_emb = image_emb.view(batch_size, n_views, -1)  # B x V x EmbeddingDim

        compound_emb = self.molecule_encoder(compound)  # B x CompoundDim
        compound_emb = self.molecule_projection_head(compound_emb)  # B x E
        compound_emb = F.normalize(compound_emb, dim=-1)

        loss = self.criterion(image_emb, compound_emb)

        return {"loss": loss, "image_emb": image_emb, "compound_emb": compound_emb}

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        super().on_train_start()

    def model_step(self, batch: Any, batch_idx: int, stage: str, **kwargs):
        image_emb = self.image_encoder(batch["image"])  # BatchSize x Views x ImagesDim
        image_emb = self.image_projection_head(image_emb)  # B x V x EmbeddingDim
        image_emb = F.normalize(image_emb, dim=-1)

        compound_emb = self.molecule_encoder(batch["compound"])  # B x CompoundDim
        compound_emb = self.molecule_projection_head(compound_emb)  # B x E
        compound_emb = F.normalize(compound_emb, dim=-1)

        batch_size, n_views, metric_dim = image_emb.shape

        total_image_emb = self.all_gather(image_emb, sync_grads=True)
        total_image_emb = total_image_emb.view(-1, n_views, metric_dim)
        total_compound_emb = self.all_gather(compound_emb, sync_grads=True)
        total_compound_emb = total_compound_emb.view(-1, metric_dim)

        batch_size, n_views, metric_dim = total_image_emb.shape

        losses = self.criterion(total_image_emb, total_compound_emb)

        if isinstance(losses, dict):
            loss = losses.pop("loss")
            self.log_dict(
                {f"{stage}/{k}": v for k, v in losses.items()},
                batch_size=batch_size,
                prog_bar=False,
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True,
            )
        else:
            loss = losses

        self.loss_dict[stage](loss)
        self.log(f"{stage}/loss", self.loss_dict[stage], batch_size=batch_size, sync_dist=True, **kwargs)

        if not torch.isfinite(loss):
            logger.info(f"Loss of batch {batch_idx} is {loss}. Returning None.")
            return None

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch, batch_idx, stage="train", on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch, batch_idx, stage="val", on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch, batch_idx, stage="test", on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def lr_scheduler_step(self, scheduler, metrics):
        super().lr_scheduler_step(scheduler, metrics)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            [
                {
                    "params": filter(lambda p: p.requires_grad, self.parameters()),
                    "lr": self.lr,
                }
            ]
        )

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)

            lr_scheduler_dict = {
                "scheduler": scheduler,
                "monitor": self.monitor,
                "interval": self.interval,
                "frequency": self.frequency,
                "strict": True,
                "name": "pretraining/lr",
            }

            if isinstance(scheduler, WarmUpWrapper) and isinstance(scheduler.wrapped_scheduler, ReduceLROnPlateau):
                lr_scheduler_dict["reduce_on_plateau"] = True

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_dict,
            }

        return {"optimizer": optimizer}
