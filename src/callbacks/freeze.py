"""Callbacks for freezing and unfreezing layers in a model."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


def multiplicative_func(a0: float) -> Callable[[int, float], float]:
    return lambda _, lr: a0 * lr


def additive_func(a0: float) -> Callable[[int, float], float]:
    return lambda _, lr: a0 + lr


def linear_func(a1: float, a0: float) -> Callable[[int, float], float]:
    """Return a function of the form a1 * x + a0 where x is the number of epochs."""
    return lambda epoch, lr: a1 * epoch + a0 * lr


def _get_layer(pl_module, layer_names):
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    layer = pl_module
    for name in layer_names:
        layer = getattr(layer, name)
    return layer


def _has_layer(pl_module, layer_names):
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    layer = pl_module
    for name in layer_names:
        if not hasattr(layer, name):
            return False
        layer = getattr(layer, name)
    return True


default_image_backbone = ["image_encoder", "backbone"]
default_image_entry = ["image_encoder", "entry"]
default_molecule_backbone = ["molecule_encoder", "backbone"]
default_lambda_func = multiplicative_func(1.5)


class BackboneFinetuningFromName(BaseFinetuning):
    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        backbone_name: Union[str, List[str]] = "molecule_encoder",
        group_name: Optional[str] = None,
        lambda_func: Callable = default_lambda_func,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        super().__init__()

        self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
        self.backbone_name = backbone_name
        self.group_name = group_name
        self.lambda_func: Callable[[int, float], float] = lambda_func
        self.backbone_initial_ratio_lr: float = backbone_initial_ratio_lr
        self.backbone_initial_lr: Optional[float] = backbone_initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_backbone_lr: Optional[float] = None

        self.is_aligned: bool = False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
            "is_aligned": self.is_aligned,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_backbone_lr = state_dict["previous_backbone_lr"]
        self.is_aligned = state_dict["is_aligned"]
        super().load_state_dict(state_dict)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if _has_layer(pl_module, self.backbone_name) and isinstance(_get_layer(pl_module, self.backbone_name), Module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(_get_layer(pl_module, self.backbone_name), train_bn=self.train_bn)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                _get_layer(pl_module, self.backbone_name),
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
                name=self.group_name,
            )

            logger.info(
                f"Current lr: {round(current_lr, self.rounding)}, "
                f"Backbone lr: {round(initial_backbone_lr, self.rounding)}"
            )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_diff = epoch - self.unfreeze_backbone_at_epoch
            next_backbone_lr = self.lambda_func(epoch_diff, self.previous_backbone_lr)

            if self.is_aligned or next_backbone_lr > current_lr:
                self.is_aligned = True
            else:
                self.is_aligned = False

            self.previous_backbone_lr = current_lr if (self.should_align and self.is_aligned) else next_backbone_lr

            for param_group in optimizer.param_groups:
                if param_group.get("name") == self.group_name:
                    param_group["lr"] = self.previous_backbone_lr

            if self.verbose:
                logger.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(self.previous_backbone_lr, self.rounding)}"
                )

    @staticmethod
    def unfreeze_and_add_param_group(
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        name: Optional[str] = None,
    ) -> None:
        BaseFinetuning.make_trainable(modules)
        params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.0
        params = BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            new_lr = params_lr / denom_lr
            optimizer.add_param_group({"params": params, "lr": new_lr, "name": name, "swa_lr": new_lr})


class JUMPCLFreezer(BaseFinetuning):
    def __init__(
        self,
        unfreeze_image_backbone_at_epoch: int = 5,
        unfreeze_molecule_backbone_at_epoch: int = 3,
        image_backbone: Union[str, List[str]] = default_image_backbone,
        image_entry: Union[str, List[str]] = default_image_entry,
        image_encoder_lr: Optional[float] = None,
        image_initial_denom_lr: Optional[float] = None,
        image_lambda_func: Callable[[int], float] = default_lambda_func,
        image_should_align: bool = True,
        molecule_backbone: Union[str, List[str]] = default_molecule_backbone,
        molecule_encoder_lr: Optional[float] = None,
        molecule_initial_denom_lr: Optional[float] = None,
        molecule_lambda_func: Callable[[int], float] = default_lambda_func,
        molecule_should_align: bool = True,
        train_bn: bool = False,
        verbose: bool = False,
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
            image_entry: (Union[str, List[str]], optional):
            image_encoder_lr (Optional[float], optional):
                Learning rate for the image backbone param group.
            image_initial_denom_lr (Optional[float], optional):
                If no lr is provided, the learning from the first param group will be used and divided by initial_denom_lr.
                Defaults to None.
            image_lambda_func (Callable[[int, float], float], optional):
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
            molecule_lambda_func (Callable[[int, float], float], optional):
                Function that takes the current epoch as input and returns a float. This float will be used to multiply
                the learning rate of the molecule backbone param group. Defaults to lambda x: 1.2.
            molecule_should_align (bool, optional):
                If True, the molecule backbone param group will be aligned with the first param group.
            train_bn (bool, optional):
                If True, the batch norm layers will be trained. Defaults to False.
            verbose (bool, optional):
                If True, will print the param groups and the learning rates in the logger. Defaults to False.

        Raises:
            MisconfigurationException:
                If the LightningModule does not have the required backbone modules.
        """

        super().__init__()
        self.unfreeze_image_backbone_at_epoch: int = unfreeze_image_backbone_at_epoch
        self.unfreeze_molecule_backbone_at_epoch: int = unfreeze_molecule_backbone_at_epoch

        self.image_backbone: Union[str, List[str]] = image_backbone
        self.image_entry: Optional[Union[str, List[str]]] = image_entry
        self.image_encoder_lr: Optional[float] = image_encoder_lr
        self.image_initial_denom_lr: Optional[float] = image_initial_denom_lr
        self.image_lambda_func: Callable[[int, float], float] = image_lambda_func
        self.image_should_align: bool = image_should_align

        self.molecule_backbone: Union[str, List[str]] = molecule_backbone
        self.molecule_encoder_lr: Optional[float] = molecule_encoder_lr
        self.molecule_initial_denom_lr: Optional[float] = molecule_initial_denom_lr
        self.molecule_lambda_func: Callable[[int, float], float] = molecule_lambda_func
        self.molecule_should_align: bool = molecule_should_align

        self.previous_image_backbone_lr: Optional[float] = None
        self.previous_molecule_backbone_lr: Optional[float] = None

        self.image_is_aligned: bool = False
        self.molecule_is_aligned: bool = False

        self.train_bn: bool = train_bn

        self.verbose: bool = verbose

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_image_backbone_lr": self.previous_image_backbone_lr,
            "image_is_aligned": self.image_is_aligned,
            "previous_molecule_backbone_lr": self.previous_molecule_backbone_lr,
            "molecule_is_aligned": self.molecule_is_aligned,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_image_backbone_lr = state_dict["previous_image_backbone_lr"]
        self.previous_molecule_backbone_lr = state_dict["previous_molecule_backbone_lr"]
        super().load_state_dict(state_dict)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Check if the model has an image encoder
        try:
            logger.debug("Loading image backbone")
            image_backbone = _get_layer(pl_module, self.image_backbone)
            logger.info(f"{len(list(image_backbone.parameters()))} parameters in image backbone")
        except AttributeError:
            raise MisconfigurationException("The LightningModule does not have a valid image backbone")

        # Check if the model has a molecule encoder
        try:
            logger.debug("Loading molecule backbone")
            molecule_backbone = _get_layer(pl_module, self.molecule_backbone)
            logger.info(f"{len(list(molecule_backbone.parameters()))} parameters in molecule backbone")
        except AttributeError:
            raise MisconfigurationException("The LightningModule does not have a valid molecule backbone")

        # return super().on_fit_start(trainer, pl_module)  # TODO: check if this is needed and debug for lr finder

    def log_message(self, message):
        if self.verbose:
            logger.info(message)
        else:
            logger.debug(message)

    def freeze_before_training(self, pl_module):
        """Freeze layers before training."""
        if self.unfreeze_image_backbone_at_epoch > 0:
            self.log_message("Freezing image encoder")
            self.freeze(_get_layer(pl_module, self.image_backbone), train_bn=self.train_bn)

            if self.image_entry:
                self.log_message("Unfreezing image entry")
                self.make_trainable(_get_layer(pl_module, self.image_entry))

        if self.unfreeze_molecule_backbone_at_epoch > 0:
            self.log_message("Freezing molecule encoder")
            self.freeze(_get_layer(pl_module, self.molecule_backbone), train_bn=self.train_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """When unfreeze epoch is reached, unfreeze the layers and add the
        param group to the optimizer."""

        # Unfreezes the image encoder and adds the param group to the optimizer
        if current_epoch == self.unfreeze_image_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            self.previous_image_backbone_lr = self.image_encoder_lr or current_lr / self.image_initial_denom_lr

            self.log_message(f"Unfreezing image encoder with lr {self.previous_image_backbone_lr}")
            self.unfreeze_and_add_param_group(
                modules=_get_layer(pl_module, self.image_backbone),
                optimizer=optimizer,
                train_bn=self.train_bn,
                lr=self.previous_image_backbone_lr,
                initial_denom_lr=self.image_initial_denom_lr,
                name="image_encoder_unfrozen",
            )
        elif current_epoch > self.unfreeze_image_backbone_at_epoch:
            next_image_backbone_lr, image_is_aligned = self.update_lr(
                optimizer=optimizer,
                diff_epoch=current_epoch - self.unfreeze_image_backbone_at_epoch,
                previous_lr=self.previous_image_backbone_lr,
                lambda_func=self.image_lambda_func,
                should_align=self.image_should_align,
                is_aligned=self.image_is_aligned,
                name="image_encoder_unfrozen",
            )
            self.previous_image_backbone_lr = next_image_backbone_lr
            self.image_is_aligned = image_is_aligned

        # Unfreezes the molecule encoder and adds the param group to the optimizer
        if current_epoch == self.unfreeze_molecule_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            self.previous_molecule_backbone_lr = self.molecule_encoder_lr or current_lr / self.molecule_initial_denom_lr

            self.log_message(f"Unfreezing molecule encoder with lr {self.previous_molecule_backbone_lr}")
            self.unfreeze_and_add_param_group(
                modules=_get_layer(pl_module, self.molecule_backbone),
                optimizer=optimizer,
                train_bn=self.train_bn,
                lr=self.previous_molecule_backbone_lr,
                initial_denom_lr=self.molecule_initial_denom_lr,
                name="molecule_encoder_unfrozen",
            )
        elif current_epoch > self.unfreeze_molecule_backbone_at_epoch:
            next_molecule_backbone_lr, molecule_is_aligned = self.update_lr(
                optimizer=optimizer,
                diff_epoch=current_epoch - self.unfreeze_molecule_backbone_at_epoch,
                previous_lr=self.previous_molecule_backbone_lr,
                lambda_func=self.molecule_lambda_func,
                should_align=self.molecule_should_align,
                is_aligned=self.molecule_is_aligned,
                name="molecule_encoder_unfrozen",
            )
            self.previous_molecule_backbone_lr = next_molecule_backbone_lr
            self.molecule_is_aligned = molecule_is_aligned

        if (
            current_epoch >= self.unfreeze_image_backbone_at_epoch
            and current_epoch >= self.unfreeze_molecule_backbone_at_epoch
        ):
            current_lr = optimizer.param_groups[0]["lr"]
            if self.previous_image_backbone_lr < current_lr or self.previous_molecule_backbone_lr < current_lr:
                self.log_message(
                    f"Current lr: {current_lr:.4f}, "
                    f"Image backbone lr: {self.previous_image_backbone_lr:.4f}, "
                    f"Molecule backbone lr: {self.previous_molecule_backbone_lr:.4f}"
                )

    @staticmethod
    def update_lr(
        optimizer: Optimizer,
        diff_epoch: int,
        previous_lr: float,
        lambda_func: Callable[[int, float], float],
        should_align: bool,
        is_aligned: bool,
        name: str,
    ):
        current_lr = optimizer.param_groups[0]["lr"]
        next_backbone_lr = lambda_func(diff_epoch, previous_lr)

        if is_aligned or next_backbone_lr > current_lr:
            local_is_aligned = True
        else:
            local_is_aligned = False

        next_backbone_lr = current_lr if (should_align and local_is_aligned) else next_backbone_lr

        for param_group in optimizer.param_groups:
            if param_group.get("name") == name:
                param_group["lr"] = next_backbone_lr

        return next_backbone_lr, local_is_aligned

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
            new_lr = params_lr / denom_lr
            optimizer.add_param_group({"params": params, "lr": new_lr, "name": name, "swa_lr": new_lr})
