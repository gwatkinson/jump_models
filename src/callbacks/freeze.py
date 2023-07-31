"""Callbacks for freezing and unfreezing layers in a model."""

import logging
from typing import Optional

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


class JUMPCLFreezer(BaseFinetuning):
    def __init__(
        self,
        unfreeze_image_encoder_at_epoch: int = 5,
        unfreeze_molecule_encoder_at_epoch: int = 3,
        image_encoder_lr: Optional[float] = 1e-4,
        image_initial_denom_lr: Optional[float] = None,
        image_encoder_name: Optional[str] = "image_encoder",
        image_base_model_name: Optional[str] = "model",
        image_head_name: Optional[str] = "myfc",
        molecule_encoder_lr: Optional[float] = 1e-4,
        molecule_initial_denom_lr: Optional[float] = None,
        molecule_encoder_name: Optional[str] = "molecule_encoder",
        molecule_base_model_name: Optional[str] = "base_model",
        molecule_head_name: Optional[str] = "head",
    ):
        """Freeze and unfreeze layers in a cross modal model.

        This callback allows to freeze the image and molecule backbone at the beginning of the training and unfreeze
        them at a given epoch, with a different learning rate if needed.

        Args:
            unfreeze_image_encoder_at_epoch (int, optional):
                Epoch at which the image encoder will be unfrozen. Defaults to 5.
            unfreeze_molecule_encoder_at_epoch (int, optional):
                Epoch at which the molecule encoder will be unfrozen. Defaults to 3.
            image_encoder_lr (Optional[float], optional):
                Learning rate for the image backbone param group.
            image_initial_denom_lr (Optional[float], optional):
                If no lr is provided, the learning from the first param group will be used and divided by initial_denom_lr.
                Defaults to None.
            image_encoder_name (Optional[str], optional):
                Name of the image encoder module in the LightningModule. Defaults to "image_encoder".
            image_base_model_name (Optional[str], optional):
                Name of the image base model module in the LightningModule. Defaults to "model".
            image_head_name (Optional[str], optional):
                Name of the image head module in the LightningModule. Defaults to "myfc".
            molecule_encoder_lr (Optional[float], optional):
                Learning rate for the molecule backbone param group.
            molecule_initial_denom_lr (Optional[float], optional):
                If no lr is provided, the learning from the first param group will be used and divided by initial_denom_lr.
                Defaults to None.
            molecule_encoder_name (Optional[str], optional):
                Name of the molecule encoder module in the LightningModule. Defaults to "molecule_encoder".
            molecule_base_model_name (Optional[str], optional):
                Name of the molecule base model module in the LightningModule. Defaults to "base_model".
            molecule_head_name (Optional[str], optional):
                Name of the molecule head module in the LightningModule. Defaults to "head".

        Raises:
            MisconfigurationException:
                If the LightningModule does not have the required backbone modules.
        """

        super().__init__()
        self.unfreeze_image_encoder_at_epoch = unfreeze_image_encoder_at_epoch
        self.unfreeze_molecule_encoder_at_epoch = unfreeze_molecule_encoder_at_epoch
        self.molecule_encoder_lr = molecule_encoder_lr
        self.image_encoder_lr = image_encoder_lr
        self.image_initial_denom_lr = image_initial_denom_lr
        self.molecule_initial_denom_lr = molecule_initial_denom_lr

        self.image_encoder_name = image_encoder_name
        self.molecule_encoder_name = molecule_encoder_name
        self.image_base_model_name = image_base_model_name
        self.molecule_base_model_name = molecule_base_model_name
        self.image_head_name = image_head_name
        self.molecule_head_name = molecule_head_name

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Check if the model has an image encoder
        image_encoder = hasattr(pl_module, self.image_encoder_name) and isinstance(
            getattr(pl_module, self.image_encoder_name), Module
        )
        if image_encoder:
            image_fc = hasattr(pl_module.image_encoder, self.image_head_name) and isinstance(
                getattr(pl_module.image_encoder, self.image_head_name), Module
            )
            image_model = hasattr(pl_module.image_encoder, self.image_base_model_name) and isinstance(
                getattr(pl_module.image_encoder, self.image_base_model_name), Module
            )
        else:
            image_fc = False
            image_model = False

        # Check if the model has a molecule encoder
        molecule_encoder = hasattr(pl_module, self.molecule_encoder_name) and isinstance(
            getattr(pl_module, self.molecule_encoder_name), Module
        )
        if molecule_encoder:
            molecule_fc = hasattr(pl_module.molecule_encoder, self.molecule_head_name) and isinstance(
                getattr(pl_module.molecule_encoder, self.molecule_head_name), Module
            )
            molecule_model = hasattr(pl_module.molecule_encoder, self.molecule_base_model_name) and isinstance(
                getattr(pl_module.molecule_encoder, self.molecule_base_model_name), Module
            )
        else:
            molecule_fc = False
            molecule_model = False

        self.layers = {
            "image_encoder": image_encoder,
            "image_fc": image_fc,
            "image_model": image_model,
            "molecule_encoder": molecule_encoder,
            "molecule_fc": molecule_fc,
            "molecule_model": molecule_model,
        }

        # Get the image and molecule backbones
        if image_model:
            self.image_backbone = self.image_base_model_name
        elif image_encoder:
            self.image_backbone = self.image_encoder_name
        else:
            self.image_backbone = None

        if molecule_model:
            self.molecule_backbone = self.molecule_base_model_name
        elif molecule_encoder:
            self.molecule_backbone = self.molecule_encoder_name
        else:
            self.molecule_backbone = None

        if self.image_backbone and self.molecule_backbone:
            return super().on_fit_start(trainer, pl_module)

        raise MisconfigurationException("The LightningModule does not have a valid image or molecule backbone")

    def freeze_before_training(self, pl_module):
        """Freeze layers before training."""
        if self.unfreeze_image_encoder_at_epoch > 0:
            self.freeze(getattr(pl_module, self.image_backbone))

        if self.unfreeze_molecule_encoder_at_epoch > 0:
            self.freeze(getattr(pl_module, self.molecule_backbone))

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """When unfreeze epoch is reached, unfreeze the layers and add the
        param group to the optimizer."""
        if current_epoch == self.unfreeze_image_encoder_at_epoch:
            # Unfreezes the image encoder and adds the param group to the optimizer
            self.unfreeze_and_add_param_group(
                modules=getattr(pl_module, self.image_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.image_encoder_lr,
                initial_denom_lr=self.image_initial_denom_lr,
            )

        if current_epoch == self.unfreeze_molecule_encoder_at_epoch:
            # Unfreezes the molecule encoder and adds the param group to the optimizer
            self.unfreeze_and_add_param_group(
                modules=getattr(pl_module, self.molecule_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.molecule_encoder_lr,
                initial_denom_lr=self.molecule_initial_denom_lr,
            )
