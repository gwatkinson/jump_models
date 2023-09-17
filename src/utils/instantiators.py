from copy import deepcopy
from typing import List, Optional

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig, verbose=True) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        if verbose:
            log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            if verbose:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig, verbose=True) -> List[Logger]:
    """Instantiates loggers from config."""

    logger: List[Logger] = []

    if not logger_cfg:
        if verbose:
            log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            if verbose:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_evaluator(
    evaluator_cfg: DictConfig,
    model_cfg: DictConfig,
    logger: Optional[List[Logger]] = None,
    ckpt_path: Optional[str] = None,
    strict: bool = True,
    name: Optional[str] = None,
):
    model_cfg = deepcopy(model_cfg)

    if ckpt_path is not None:
        model_cfg["_target_"] += ".load_from_checkpoint"
        with open_dict(model_cfg):
            model_cfg["checkpoint_path"] = ckpt_path
            model_cfg["map_location"] = "cpu"
            model_cfg["strict"] = strict

    if isinstance(evaluator_cfg, DictConfig):
        if "model" in evaluator_cfg:
            model = hydra.utils.instantiate(model_cfg)
            module = hydra.utils.instantiate(evaluator_cfg.model, cross_modal_module=model)
        else:
            raise ValueError("Evaluator config must contain a model!")

        if "datamodule" in evaluator_cfg:
            datamodule = hydra.utils.instantiate(evaluator_cfg.datamodule)
        else:
            raise ValueError("Evaluator config must contain a datamodule!")

        if "callbacks" in evaluator_cfg:
            callbacks = instantiate_callbacks(evaluator_cfg.callbacks, verbose=False)
        else:
            callbacks = []

        if "trainer" in evaluator_cfg:
            trainer = hydra.utils.instantiate(evaluator_cfg.trainer, callbacks=callbacks, logger=logger)
        else:
            trainer = None

        if evaluator_cfg.evaluator.name is None:
            evaluator_cfg.evaluator.name = name

        log.info(f"Instantiating evaluator <{evaluator_cfg.model._target_}>")
        evaluator = hydra.utils.instantiate(
            evaluator_cfg.evaluator, model=module, datamodule=datamodule, trainer=trainer
        )

        return evaluator


def instantiate_evaluator_list(
    evaluator_list_cfg: DictConfig,
    model_cfg: DictConfig,
    logger: Optional[List[Logger]] = None,
    ckpt_path: Optional[str] = None,
    name: Optional[str] = None,
):
    """Instantiates evaluator list from config."""

    from src.eval.evaluators import Evaluator, EvaluatorList

    evaluators: List[Evaluator] = []
    model_cfg = deepcopy(model_cfg)

    if not evaluator_list_cfg:
        log.warning("No evaluator configs found! Skipping...")
        return EvaluatorList(evaluators=evaluators)

    if not isinstance(evaluator_list_cfg, DictConfig):
        raise TypeError("Evaluator config must be a DictConfig!")

    if ckpt_path is not None:
        model_cfg["_target_"] += ".load_from_checkpoint"
        with open_dict(model_cfg):
            model_cfg["checkpoint_path"] = ckpt_path

    for evaluator_name, evaluator_cfg in evaluator_list_cfg.items():
        if isinstance(evaluator_cfg, DictConfig):
            if "model" in evaluator_cfg:
                model = hydra.utils.instantiate(model_cfg)
                module = hydra.utils.instantiate(evaluator_cfg.model, cross_modal_module=model)
            else:
                raise ValueError("Evaluator config must contain a model!")

            if "datamodule" in evaluator_cfg:
                datamodule = hydra.utils.instantiate(evaluator_cfg.datamodule)
            else:
                raise ValueError("Evaluator config must contain a datamodule!")

            if "callbacks" in evaluator_cfg:
                callbacks = instantiate_callbacks(evaluator_cfg.callbacks, verbose=False)
            else:
                callbacks = []

            if "trainer" in evaluator_cfg:
                trainer = hydra.utils.instantiate(evaluator_cfg.trainer, callbacks=callbacks, logger=logger)
            else:
                trainer = None

            if evaluator_cfg.evaluator.name is None:
                evaluator_cfg.evaluator.name = evaluator_name

            log.info(f"Instantiating evaluator <{evaluator_cfg.model._target_}>")
            evaluator = hydra.utils.instantiate(
                evaluator_cfg.evaluator, model=module, datamodule=datamodule, trainer=trainer
            )

            evaluators.append(evaluator)

    return EvaluatorList(evaluators=evaluators, name=name)
