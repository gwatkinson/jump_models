from typing import List, Optional

import hydra
import torch.nn as nn
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""

    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_evaluator_list(
    evaluator_list_cfg: DictConfig,
    cross_modal_module: nn.Module,
    logger: Optional[List[Logger]] = None,
    ckpt_path: Optional[str] = None,
    name: Optional[str] = None,
):
    """Instantiates evaluator list from config."""

    from src.eval.evaluators import Evaluator, EvaluatorList

    evaluators: List[Evaluator] = []

    if not evaluator_list_cfg:
        log.warning("No evaluator configs found! Skipping...")
        return EvaluatorList(evaluators=evaluators)

    if not isinstance(evaluator_list_cfg, DictConfig):
        raise TypeError("Evaluator config must be a DictConfig!")

    for ev_name, ev_conf in evaluator_list_cfg.items():
        if isinstance(ev_conf, DictConfig):
            if "model" in ev_conf:
                module = hydra.utils.instantiate(ev_conf.model, cross_modal_module=cross_modal_module)
            else:
                raise ValueError("Evaluator config must contain a model!")

            if "datamodule" in ev_conf:
                datamodule = hydra.utils.instantiate(ev_conf.datamodule)
            else:
                raise ValueError("Evaluator config must contain a datamodule!")

            if "callbacks" in ev_conf:
                callbacks = instantiate_callbacks(ev_conf.callbacks)
            else:
                callbacks = []

            if "trainer" in ev_conf:
                trainer = hydra.utils.instantiate(ev_conf.trainer, callbacks=callbacks, logger=logger)
            else:
                trainer = None

            if ev_conf.evaluator.name is None:
                ev_conf.evaluator.name = ev_name

            log.info(f"Instantiating evaluator <{ev_conf.model._target_}>")
            evaluator = hydra.utils.instantiate(ev_conf.evaluator, model=module, datamodule=datamodule, trainer=trainer)

            evaluators.append(evaluator)

    return EvaluatorList(evaluators=evaluators, name=name)
