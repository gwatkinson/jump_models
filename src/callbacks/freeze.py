"""Callbacks for freezing and unfreezing layers in a model."""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim.optimizer import Optimizer

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


def multiplicative_func(a0: float) -> Callable[[int], float]:
    return lambda _: a0


def linear_func(a1: float, a0: float) -> Callable[[int], float]:
    """Return a function of the form a1 * x + a0 where x is the number of epochs."""
    return lambda x: a1 * x + a0


default_image_backbone = ["image_encoder", "backbone"]
default_molecule_backbone = ["molecule_encoder", "backbone"]
default_lambda_func = multiplicative_func(1.5)


class JUMPCLFreezer(BaseFinetuning):
    def __init__(
        self,
        unfreeze_image_backbone_at_epoch: int = 5,
        unfreeze_molecule_backbone_at_epoch: int = 3,
        image_backbone: Union[str, List[str]] = default_image_backbone,
        image_encoder_lr: Optional[float] = 1e-4,
        image_initial_denom_lr: Optional[float] = None,
        image_lambda_func: Callable[[int], float] = default_lambda_func,
        image_should_align: bool = True,
        molecule_backbone: Union[str, List[str]] = default_molecule_backbone,
        molecule_encoder_lr: Optional[float] = 1e-4,
        molecule_initial_denom_lr: Optional[float] = None,
        molecule_lambda_func: Callable[[int], float] = default_lambda_func,
        molecule_should_align: bool = True,
    ):
        """Freeze and unfreeze layers in a cross modal model.

        This callback allows to freeze the image and molecule backbone at the beginning of the training and unfreeze
        them at a given epoch, with a different learning rate if needed.

        Args:
            unfreeze_image_backbone_at_epoch (int, optional):
                Epoch at which the image encoder will be unfrozen. Defaults to 5.
            unfreeze_molecule_backbone_at_epoch (int, optional):
                Epoch at which the molecule encoder will be unfrozen. Defaults to 3.
            image_backbone (Union[str, List[str]], optional):
                Name of the image backbone modules. If a list is given, the param groups will be created in the order
                of the list. Defaults to ["image_encoder", "model"].
            image_encoder_lr (Optional[float], optional):
                Learning rate for the image backbone param group.
            image_initial_denom_lr (Optional[float], optional):
                If no lr is provided, the learning from the first param group will be used and divided by initial_denom_lr.
                Defaults to None.
            image_lambda_func (Callable[[int], float], optional):
                Function that takes the current epoch as input and returns a float. This float will be used to multiply
                the learning rate of the image backbone param group. Defaults to lambda x: 1.2.
            image_should_align (bool, optional):
                If True, the image backbone param group will be aligned with the first param group.
            molecule_backbone (Union[str, List[str]], optional):
                Name of the molecule backbone modules. If a list is given, the param groups will be created in the order
                of the list. Defaults to ["molecule_encoder", "base_model"].
            molecule_encoder_lr (Optional[float], optional):
                Learning rate for the molecule backbone param group.
            molecule_initial_denom_lr (Optional[float], optional):
                If no lr is provided, the learning from the first param group will be used and divided by initial_denom_lr.
                Defaults to None.
            molecule_lambda_func (Callable[[int], float], optional):
                Function that takes the current epoch as input and returns a float. This float will be used to multiply
                the learning rate of the molecule backbone param group. Defaults to lambda x: 1.2.
            molecule_should_align (bool, optional):
                If True, the molecule backbone param group will be aligned with the first param group.

        Raises:
            MisconfigurationException:
                If the LightningModule does not have the required backbone modules.
        """

        super().__init__()
        self.unfreeze_image_backbone_at_epoch = unfreeze_image_backbone_at_epoch
        self.unfreeze_molecule_backbone_at_epoch = unfreeze_molecule_backbone_at_epoch

        self.image_backbone = image_backbone
        self.image_encoder_lr = image_encoder_lr
        self.image_initial_denom_lr = image_initial_denom_lr
        self.image_lambda_func = image_lambda_func
        self.image_should_align = image_should_align

        self.molecule_backbone = molecule_backbone
        self.molecule_encoder_lr = molecule_encoder_lr
        self.molecule_initial_denom_lr = molecule_initial_denom_lr
        self.molecule_lambda_func = molecule_lambda_func
        self.molecule_should_align = molecule_should_align

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_image_backbone_lr": self.previous_image_backbone_lr,
            "previous_molecule_backbone_lr": self.previous_molecule_backbone_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_image_backbone_lr = state_dict["previous_image_backbone_lr"]
        self.previous_molecule_backbone_lr = state_dict["previous_molecule_backbone_lr"]
        super().load_state_dict(state_dict)

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

        logger.debug(f"Params group of optimizer 0: {trainer.optimizers[0].param_groups}")

        optimizer = trainer.optimizers[0]
        param_groups = optimizer.param_groups

        id_to_param_group_id = defaultdict(list)

        for i, group in enumerate(param_groups):
            for p in group["params"]:
                id_to_param_group_id[id(p)].append(i)
        multi_group_params = {k: v for k, v in id_to_param_group_id.items() if len(v) > 1}

        logger.debug(f"Parameter in multiple groups:\n{multi_group_params}")
        logger.debug(f"Param Id to parameter groups:\n{id_to_param_group_id}")

        # return super().on_fit_start(trainer, pl_module)  # TODO: check if this is needed and debug for lr finder

    def freeze_before_training(self, pl_module):
        """Freeze layers before training."""
        if self.unfreeze_image_backbone_at_epoch > 0:
            logger.info("Freezing image encoder")
            self.freeze(self._get_backbone(pl_module, self.image_backbone))

        if self.unfreeze_molecule_backbone_at_epoch > 0:
            logger.info("Freezing molecule encoder")
            self.freeze(self._get_backbone(pl_module, self.molecule_backbone))

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """When unfreeze epoch is reached, unfreeze the layers and add the
        param group to the optimizer."""

        # Unfreezes the image encoder and adds the param group to the optimizer
        if current_epoch == self.unfreeze_image_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            self.previous_image_backbone_lr = self.image_encoder_lr or current_lr

            logger.info(f"Unfreezing image encoder with lr {self.previous_image_backbone_lr}")
            self.unfreeze_and_add_param_group(
                modules=self._get_backbone(pl_module, self.image_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.previous_image_backbone_lr,
                initial_denom_lr=self.image_initial_denom_lr,
                name="image_encoder",
            )
        elif current_epoch > self.unfreeze_image_backbone_at_epoch:
            next_image_backbone_lr = self.update_lr(
                optimizer=optimizer,
                epoch=current_epoch - self.unfreeze_image_backbone_at_epoch,
                previous_backbone_lr=self.previous_image_backbone_lr,
                lambda_func=self.image_lambda_func,
                should_align=self.image_should_align,
                name="image_encoder",
            )
            self.previous_image_backbone_lr = next_image_backbone_lr

        # Unfreezes the molecule encoder and adds the param group to the optimizer
        if current_epoch == self.unfreeze_molecule_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            self.previous_molecule_backbone_lr = self.molecule_encoder_lr or current_lr

            logger.info(f"Unfreezing molecule encoder with lr {self.previous_molecule_backbone_lr}")
            self.unfreeze_and_add_param_group(
                modules=self._get_backbone(pl_module, self.molecule_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.previous_molecule_backbone_lr,
                initial_denom_lr=self.molecule_initial_denom_lr,
                name="molecule_encoder",
            )
        elif current_epoch > self.unfreeze_molecule_backbone_at_epoch:
            next_molecule_backbone_lr = self.update_lr(
                optimizer=optimizer,
                epoch=current_epoch - self.unfreeze_molecule_backbone_at_epoch,
                previous_backbone_lr=self.previous_molecule_backbone_lr,
                lambda_func=self.molecule_lambda_func,
                should_align=self.molecule_should_align,
                name="molecule_encoder",
            )
            self.previous_molecule_backbone_lr = next_molecule_backbone_lr

        if (
            current_epoch >= self.unfreeze_image_backbone_at_epoch
            and current_epoch >= self.unfreeze_molecule_backbone_at_epoch
        ):
            current_lr = optimizer.param_groups[0]["lr"]
            if self.previous_image_backbone_lr < current_lr or self.previous_molecule_backbone_lr < current_lr:
                logger.info(
                    f"Current lr: {current_lr:.4f}, "
                    f"Image backbone lr: {self.previous_image_backbone_lr:.4f}, "
                    f"Molecule backbone lr: {self.previous_molecule_backbone_lr:.4f}"
                )

    @staticmethod
    def update_lr(
        optimizer: Optimizer,
        diff_epoch: int,
        previous_lr: float,
        lambda_func: Callable[[int], float],
        should_align: bool,
        name: str,
    ):
        current_lr = optimizer.param_groups[0]["lr"]
        next_backbone_lr = lambda_func(diff_epoch) * previous_lr
        next_backbone_lr = current_lr if (should_align and next_backbone_lr > current_lr) else next_backbone_lr

        for param_group in optimizer.param_groups:
            if param_group.get("name") == name:
                param_group["lr"] = next_backbone_lr

        return next_backbone_lr

    @staticmethod
    def unfreeze_and_add_param_group(
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """Unfreezes a module and adds its parameters to an optimizer.

        Args:
            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.
            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`
            lr: Learning rate for the new param group.
            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by `initial_denom_lr`.
            train_bn: Whether to train the BatchNormalization layers.
            name: Name of the param group.If None, the name will not be added to the param group.
        """
        BaseFinetuning.make_trainable(modules)
        params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.0
        params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            updated_params = {"params": params, "lr": params_lr / denom_lr}
            if name is not None:
                updated_params["name"] = name

            optimizer.add_param_group(updated_params)
