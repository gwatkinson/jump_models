from typing import Any, Optional

import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric

from src.modules.lr_schedulers.warmup_wrapper import WarmUpWrapper
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class BasicJUMPModule(LightningModule):
    """Basic Jump LightningModule to run a simple contrastive training.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

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
        image_backbone: str = "backbone",
        image_head: str = "projection_head",
        molecule_backbone: str = "backbone",
        molecule_head: str = "projection_head",
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

        if image_backbone is not None:
            self.image_backbone = getattr(self.image_encoder, image_backbone)
        else:
            self.image_backbone = None
        if molecule_backbone is not None:
            self.molecule_backbone = getattr(self.molecule_encoder, molecule_backbone)
        else:
            self.molecule_backbone = None

        if image_head is not None:
            self.image_head = getattr(self.image_encoder, image_head)
        else:
            self.image_head = None
        if molecule_head is not None:
            self.molecule_head = getattr(self.molecule_encoder, molecule_head)
        else:
            self.molecule_head = None

        # embedding dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lr = lr

        # self.steps = {
        #     "train": 0.0,
        #     "val": 0.0,
        #     "test": 0.0,
        # }

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

    def forward(self, image, compound):
        image_emb = self.image_encoder(image)  # BxE
        compound_emb = self.molecule_encoder(compound)  # BxE

        return {"image_emb": image_emb, "compound_emb": compound_emb}

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        super().on_train_start()

    def model_step(self, batch: Any, batch_idx: int, stage: str, **kwargs):
        image_emb = self.image_encoder(batch["image"])
        compound_emb = self.molecule_encoder(batch["compound"])
        batch_size = image_emb.shape[0]

        losses = self.criterion(image_emb, compound_emb)
        if isinstance(losses, dict):
            loss = losses.pop("loss")
            self.log_dict(
                {f"{stage}/{k}": v for k, v in losses.items()},
                batch_size=batch_size,
                prog_bar=False,
                on_step=(stage == "train"),
                on_epoch=True,
            )
        else:
            loss = losses

        self.loss_dict[stage](loss)
        self.log(f"{stage}/loss", self.loss_dict[stage], batch_size=batch_size, **kwargs)

        # self.steps[stage] += 1
        # self.log(f"{stage}/steps", self.steps[stage], prog_bar=False, on_epoch=False, on_step=True)

        if stage == "train":
            try:
                temperature = self.criterion.temperature.value.item()
                self.log("model/temperature", temperature, prog_bar=False, on_epoch=True, on_step=False)
            except AttributeError:
                pass

        if self.trainer.num_devices > 1:
            devices_loss = self.all_gather(loss)  # Need to gather losses when using DDP to not fall out of sync
            condition = all(torch.isfinite(loss) for loss in devices_loss)
        else:
            condition = torch.isfinite(loss)

        if not condition:
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
        params_groups = [
            {
                "params": list(self.criterion.parameters()),
                "name": "criterion",
            }
        ]

        if self.image_head is not None and self.image_backbone is not None:
            params_groups.append(
                {
                    "params": list(self.image_head.parameters()),
                    "name": "image_projection_head",
                }
            )
            params_groups.append(
                {
                    "params": list(self.image_backbone.parameters()),
                    "name": "image_encoder",
                }
            )
        else:
            params_groups.append(
                {
                    "params": list(self.image_encoder.parameters()),
                    "name": "image_encoder",
                }
            )

        if self.molecule_head is not None and self.molecule_backbone is not None:
            params_groups.append(
                {
                    "params": list(self.molecule_head.parameters()),
                    "name": "molecule_projection_head",
                }
            )
            params_groups.append(
                {
                    "params": list(self.molecule_backbone.parameters()),
                    "name": "molecule_encoder",
                }
            )
        else:
            params_groups.append(
                {
                    "params": list(self.molecule_encoder.parameters()),
                    "name": "molecule_encoder",
                }
            )

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

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)

            lr_scheduler_dict = {
                "scheduler": scheduler,
                "monitor": self.monitor,
                "interval": self.interval,
                "frequency": self.frequency,
                "strict": True,
                "name": "lr/jump_cl",
            }

            if isinstance(scheduler, WarmUpWrapper) and isinstance(scheduler.wrapped_scheduler, ReduceLROnPlateau):
                lr_scheduler_dict["reduce_on_plateau"] = True

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_dict,
            }

        return {"optimizer": optimizer}
