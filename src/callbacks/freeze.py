"""Callbacks for freezing and unfreezing layers in a model."""

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
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


def multiplicative_func(a0: float) -> Callable[[int, float], float]:
    return lambda _, lr: a0 * lr


def additive_func(a0: float) -> Callable[[int, float], float]:
    return lambda _, lr: a0 + lr


def linear_func(a1: float, a0: float) -> Callable[[int, float], float]:
    """Return a function of the form a1 * x + a0 where x is the number of epochs."""
    return lambda epoch, lr: a1 * epoch + a0 * lr


default_image_backbone = ["image_encoder", "backbone"]
default_molecule_backbone = ["molecule_encoder", "backbone"]
default_lambda_func = multiplicative_func(1.5)


class JUMPCLFreezer(BaseFinetuning):
    def __init__(
        self,
        unfreeze_image_backbone_at_epoch: int = 5,
        unfreeze_molecule_backbone_at_epoch: int = 3,
        image_backbone: Union[str, List[str]] = default_image_backbone,
        image_encoder_lr: Optional[float] = None,
        image_initial_denom_lr: Optional[float] = None,
        image_lambda_func: Callable[[int], float] = default_lambda_func,
        image_should_align: bool = True,
        molecule_backbone: Union[str, List[str]] = default_molecule_backbone,
        molecule_encoder_lr: Optional[float] = None,
        molecule_initial_denom_lr: Optional[float] = None,
        molecule_lambda_func: Callable[[int], float] = default_lambda_func,
        molecule_should_align: bool = True,
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
            verbose (bool, optional):
                If True, will print the param groups and the learning rates in the logger. Defaults to False.

        Raises:
            MisconfigurationException:
                If the LightningModule does not have the required backbone modules.
        """

        super().__init__()
        self.unfreeze_image_backbone_at_epoch: int = unfreeze_image_backbone_at_epoch
        self.unfreeze_molecule_backbone_at_epoch: int = unfreeze_molecule_backbone_at_epoch

        self.image_backbone: List[str] = image_backbone
        self.image_encoder_lr: Optional[float] = image_encoder_lr
        self.image_initial_denom_lr: Optional[float] = image_initial_denom_lr
        self.image_lambda_func: Callable[[int, float], float] = image_lambda_func
        self.image_should_align: bool = image_should_align

        self.molecule_backbone: List[str] = molecule_backbone
        self.molecule_encoder_lr: Optional[float] = molecule_encoder_lr
        self.molecule_initial_denom_lr: Optional[float] = molecule_initial_denom_lr
        self.molecule_lambda_func: Callable[[int, float], float] = molecule_lambda_func
        self.molecule_should_align: bool = molecule_should_align

        self.previous_image_backbone_lr: Optional[float] = None
        self.previous_molecule_backbone_lr: Optional[float] = None

        self.image_is_aligned: bool = False
        self.molecule_is_aligned: bool = False

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
            image_backbone = self._get_backbone(pl_module, self.image_backbone)
            logger.info(f"{len(list(image_backbone.parameters()))} parameters in image backbone")
        except AttributeError:
            raise MisconfigurationException("The LightningModule does not have a valid image backbone")

        # Check if the model has a molecule encoder
        try:
            logger.debug("Loading molecule backbone")
            molecule_backbone = self._get_backbone(pl_module, self.molecule_backbone)
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
            self.freeze(self._get_backbone(pl_module, self.image_backbone))

        if self.unfreeze_molecule_backbone_at_epoch > 0:
            self.log_message("Freezing molecule encoder")
            self.freeze(self._get_backbone(pl_module, self.molecule_backbone))

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """When unfreeze epoch is reached, unfreeze the layers and add the
        param group to the optimizer."""

        # Unfreezes the image encoder and adds the param group to the optimizer
        if current_epoch == self.unfreeze_image_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            self.previous_image_backbone_lr = self.image_encoder_lr or current_lr / self.image_initial_denom_lr

            self.log_message(f"Unfreezing image encoder with lr {self.previous_image_backbone_lr}")
            self.unfreeze_and_add_param_group(
                modules=self._get_backbone(pl_module, self.image_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.previous_image_backbone_lr,
                initial_denom_lr=self.image_initial_denom_lr,
                name="image_encoder",
            )
        elif current_epoch > self.unfreeze_image_backbone_at_epoch:
            next_image_backbone_lr, image_is_aligned = self.update_lr(
                optimizer=optimizer,
                diff_epoch=current_epoch - self.unfreeze_image_backbone_at_epoch,
                previous_lr=self.previous_image_backbone_lr,
                lambda_func=self.image_lambda_func,
                should_align=self.image_should_align,
                is_aligned=self.image_is_aligned,
                name="image_encoder",
            )
            self.previous_image_backbone_lr = next_image_backbone_lr
            self.image_is_aligned = image_is_aligned

        # Unfreezes the molecule encoder and adds the param group to the optimizer
        if current_epoch == self.unfreeze_molecule_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            self.previous_molecule_backbone_lr = self.molecule_encoder_lr or current_lr / self.molecule_initial_denom_lr

            self.log_message(f"Unfreezing molecule encoder with lr {self.previous_molecule_backbone_lr}")
            self.unfreeze_and_add_param_group(
                modules=self._get_backbone(pl_module, self.molecule_backbone),
                optimizer=optimizer,
                train_bn=True,
                lr=self.previous_molecule_backbone_lr,
                initial_denom_lr=self.molecule_initial_denom_lr,
                name="molecule_encoder",
            )
        elif current_epoch > self.unfreeze_molecule_backbone_at_epoch:
            next_molecule_backbone_lr, molecule_is_aligned = self.update_lr(
                optimizer=optimizer,
                diff_epoch=current_epoch - self.unfreeze_molecule_backbone_at_epoch,
                previous_lr=self.previous_molecule_backbone_lr,
                lambda_func=self.molecule_lambda_func,
                should_align=self.molecule_should_align,
                is_aligned=self.molecule_is_aligned,
                name="molecule_encoder",
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
        params = JUMPCLFreezer.filter_on_optimizer(optimizer, params)
        if params:
            new_lr = params_lr / denom_lr
            optimizer.add_param_group({"params": params, "lr": new_lr, "name": name, "swa_lr": new_lr})

    @staticmethod
    def filter_on_optimizer(optimizer: Optimizer, params: Iterable) -> List:
        """This function is used to exclude any parameter which already exists
        in this optimizer.

        Args:
            optimizer: Optimizer used for parameter exclusion
            params: Iterable of parameters used to check against the provided optimizer

        Returns:
            List of parameters not contained in this optimizer param groups
        """
        out_params = []
        removed_params = []
        for param in params:
            if not any(torch.equal(p, param) for group in optimizer.param_groups for p in group["params"]):
                out_params.append(param)
            else:
                removed_params.append(param)

        if removed_params:
            logger.warning(
                f"Removed {len(removed_params)} params from the optimizer. {len(out_params)} params remaining."
            )
            rank_zero_warn(
                "The provided params to be frozen already exist within another group of this optimizer."
                " Those parameters will be skipped.\n"
                "HINT: Did you init your optimizer in `configure_optimizer` as such:\n"
                f" {type(optimizer)}(filter(lambda p: p.requires_grad, self.parameters()), ...) ",
            )
        return out_params
