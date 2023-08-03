import logging
from typing import Any, List, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

logger = logging.getLogger(__name__)


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
        params_group_to_ignore: Optional[List[str]] = None,
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
        self.image_backbone = getattr(self.image_encoder, image_backbone)
        self.image_head = getattr(self.image_encoder, image_head)
        self.molecule_backbone = getattr(self.molecule_encoder, molecule_backbone)
        self.molecule_head = getattr(self.molecule_encoder, molecule_head)
        self.params_group_to_ignore = params_group_to_ignore

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
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_min = MinMetric()

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
        self.val_loss.reset()
        super().on_train_start()

    def model_step(self, batch: Any, stage: str, **kwargs):
        image_emb = self.image_encoder(batch["image"])
        compound_emb = self.molecule_encoder(batch["compound"])

        loss = self.criterion(
            embeddings_a=image_emb,
            embeddings_b=compound_emb,
        )

        self.loss_dict[stage](loss)
        # loss_to_log = self.loss_dict[stage](loss)
        self.log(f"{stage}/loss", self.loss_dict[stage], **kwargs)
        # self.log(f"{stage}/loss_to_log", loss_to_log, **kwargs)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch, stage="train", on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        logit_scale = self.criterion.logit_scale.exp().item()
        self.log("model/logit_scale", logit_scale, prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch, stage="val", on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_min.update(loss)  # update min so far val loss
        self.log("val/loss_min", self.val_loss_min.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch, stage="test", on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        params_groups = [
            {
                "params": list(filter(lambda p: p.requires_grad, self.image_head.parameters())),
                "name": "image_projection_head",
            },
            {
                "params": list(filter(lambda p: p.requires_grad, self.molecule_head.parameters())),
                "name": "molecule_projection_head",
            },
            {
                "params": list(filter(lambda p: p.requires_grad, self.criterion.parameters())),
                "name": "criterion",
            },
            {
                "params": list(filter(lambda p: p.requires_grad, self.image_backbone.parameters())),
                "name": "image_encoder",
            },
            {
                "params": list(filter(lambda p: p.requires_grad, self.molecule_backbone.parameters())),
                "name": "molecule_encoder",
            },
        ]
        group_lens = {group["name"]: len(group["params"]) for group in params_groups}

        logger.info(f"Params groups:\n{group_lens}")

        optimizer = self.optimizer(
            [
                group
                for group in params_groups
                if group_lens[group["name"]] > 0 and group["name"] not in self.params_group_to_ignore
            ],
            lr=self.lr,
        )

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.monitor,
                    "interval": self.interval,
                    "frequency": self.frequency,
                },
            }

        return {"optimizer": optimizer}
