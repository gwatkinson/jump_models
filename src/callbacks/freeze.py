"""Callbacks for freezing and unfreezing layers in a model."""

import logging
from typing import List, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.nn import Module

logger = logging.getLogger(__name__)


class OGBFreezer(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module, "molecule_encoder") and isinstance(pl_module.molecule_encoder, Module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException("The LightningModule should have a nn.Module `molecule_encoder` attribute")

    def freeze_before_training(self, pl_module):
        """Freeze layers before training."""
        self.freeze(pl_module.molecule_encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """When `current_epoch` is 10, feature_extractor will start
        training."""
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.molecule_encoder,
                optimizer=optimizer,
                train_bn=True,
            )


default_image_backbone = ["image_encoder", "model"]
default_molecule_backbone = ["molecule_encoder", "base_model"]


class JUMPCLFreezer(BaseFinetuning):
    def __init__(
        self,
        unfreeze_image_encoder_at_epoch: int = 5,
        unfreeze_molecule_encoder_at_epoch: int = 3,
        image_backbone: Union[str, List[str]] = default_image_backbone,
        image_encoder_lr: Optional[float] = 1e-4,
        image_initial_denom_lr: Optional[float] = None,
        molecule_backbone: Union[str, List[str]] = default_molecule_backbone,
        molecule_encoder_lr: Optional[float] = 1e-4,
        molecule_initial_denom_lr: Optional[float] = None,
    ):
        """Freeze and unfreeze layers in a cross modal model.

        This callback allows to freeze the image and molecule backbone at the beginning of the training and unfreeze
        them at a given epoch, with a different learning rate if needed.

        Args:
            unfreeze_image_encoder_at_epoch (int, optional):
                Epoch at which the image encoder will be unfrozen. Defaults to 5.
            unfreeze_molecule_encoder_at_epoch (int, optional):
                Epoch at which the molecule encoder will be unfrozen. Defaults to 3.
            image_backbone (Union[str, List[str]], optional):
                Name of the image backbone modules. If a list is given, the param groups will be created in the order
                of the list. Defaults to ["image_encoder", "model"].
            image_encoder_lr (Optional[float], optional):
                Learning rate for the image backbone param group.
            image_initial_denom_lr (Optional[float], optional):
                If no lr is provided, the learning from the first param group will be used and divided by initial_denom_lr.
                Defaults to None.
            molecule_backbone (Union[str, List[str]], optional):
                Name of the molecule backbone modules. If a list is given, the param groups will be created in the order
                of the list. Defaults to ["molecule_encoder", "base_model"].
            molecule_encoder_lr (Optional[float], optional):
                Learning rate for the molecule backbone param group.
            molecule_initial_denom_lr (Optional[float], optional):
                If no lr is provided, the learning from the first param group will be used and divided by initial_denom_lr.
                Defaults to None.

        Raises:
            MisconfigurationException:
                If the LightningModule does not have the required backbone modules.
        """

        super().__init__()
        self.unfreeze_image_encoder_at_epoch = unfreeze_image_encoder_at_epoch
        self.unfreeze_molecule_encoder_at_epoch = unfreeze_molecule_encoder_at_epoch

        self.image_backbone = image_backbone
        self.image_encoder_lr = image_encoder_lr
        self.image_initial_denom_lr = image_initial_denom_lr

        self.molecule_backbone = molecule_backbone
        self.molecule_encoder_lr = molecule_encoder_lr
        self.molecule_initial_denom_lr = molecule_initial_denom_lr

    @staticmethod
    def _get_backbone(pl_module, backbone_names):
        if isinstance(backbone_names, str):
            backbone_names = [backbone_names]

        backbone = pl_module
        for name in backbone_names:
            backbone = getattr(backbone, name)
        return backbone

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Check if the model has an image encoder
        try:
            logger.debug("Loading image backbone")
            self._get_backbone(pl_module, self.image_backbone)
        except AttributeError:
            raise MisconfigurationException("The LightningModule does not have a valid image backbone")

        # Check if the model has a molecule encoder
        try:
            logger.debug("Loading molecule backbone")
            self._get_backbone(pl_module, self.molecule_backbone)
        except AttributeError:
            raise MisconfigurationException("The LightningModule does not have a valid molecule backbone")

        # named_parameters = dict(pl_module.named_parameters())
        # logger.debug(f"Named parameters: {named_parameters.keys()}")

        return super().on_fit_start(trainer, pl_module)  # TODO: check if this is needed and debug

    def freeze_before_training(self, pl_module):
        """Freeze layers before training."""
        if self.unfreeze_image_encoder_at_epoch > 0:
            logger.info("Freezing image encoder")
            self.freeze(self._get_backbone(pl_module, self.image_backbone))

        if self.unfreeze_molecule_encoder_at_epoch > 0:
            logger.info("Freezing molecule encoder")
            self.freeze(self._get_backbone(pl_module, self.molecule_backbone))

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """When unfreeze epoch is reached, unfreeze the layers and add the
        param group to the optimizer."""
        if current_epoch == self.unfreeze_image_encoder_at_epoch:
            # Unfreezes the image encoder and adds the param group to the optimizer
            logger.info(f"Unfreezing image encoder with lr {self.image_encoder_lr}")
            self.unfreeze_and_add_param_group(
                modules=self._get_backbone(pl_module, self.image_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.image_encoder_lr,
                initial_denom_lr=self.image_initial_denom_lr,
            )

        if current_epoch == self.unfreeze_molecule_encoder_at_epoch:
            # Unfreezes the molecule encoder and adds the param group to the optimizer
            logger.info(f"Unfreezing molecule encoder with lr {self.molecule_encoder_lr}")
            self.unfreeze_and_add_param_group(
                modules=self._get_backbone(pl_module, self.molecule_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.molecule_encoder_lr,
                initial_denom_lr=self.molecule_initial_denom_lr,
            )
