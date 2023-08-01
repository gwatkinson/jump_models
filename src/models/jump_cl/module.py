import logging
from typing import Any, Optional

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
        example_input_path: Optional[str] = None,
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

        # embedding dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lr = lr

        # training
        self.optimizer = optimizer
        self.scheduler = scheduler

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_min = MinMetric()

        if example_input_path is not None:
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

    def model_step(self, batch: Any):
        image_emb = self.image_encoder(batch["image"])
        compound_emb = self.molecule_encoder(batch["compound"])

        loss = self.criterion(
            embeddings_a=image_emb,
            embeddings_b=compound_emb,
        )

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        logger.debug("Log training loss")
        self.train_loss.update(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        logger.debug("Run training step from validation_step()")
        loss = self.model_step(batch)

        # update and log metrics
        logger.debug("Log validation loss")
        self.val_loss.update(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_min.update(loss)  # update min so far val loss
        self.log("val/loss_min", self.val_loss_min.compute(), sync_dist=True, prog_bar=True)
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss.update(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        mb_params = self.molecule_encoder.backbone.parameters()

        require_grad = [p.requires_grad for p in mb_params]
        logger.info(f"Number of require grad parameters in image base model: {sum(require_grad)}/{len(require_grad)}")

        optimizer = self.optimizer(
            [
                {
                    "params": filter(lambda p: p.requires_grad, self.parameters()),
                    "lr": self.lr,
                    "name": "projection_head",
                }
            ],
            lr=self.lr,
        )
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
