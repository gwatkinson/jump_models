import logging
import os
from pathlib import Path
from typing import List, Tuple

import click
import hydra
import pyrootutils
from colorlog import ColoredFormatter
from hydra import compose, initialize_config_dir
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src import utils

# from src.eval import EvaluatorList

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #


LOG_LEVEL = logging.INFO
LOGFORMAT = (
    "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
)
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

log = utils.get_pylogger(__name__)
log.setLevel(LOG_LEVEL)
log.addHandler(stream)

# import colorlog

# handler = colorlog.StreamHandler()
# handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'))

# log = colorlog.getLogger(__name__)
# log.addHandler(handler)

# logging.basicConfig(level=logging.INFO)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    if not cfg.ckpt_path:
        raise ValueError("You need to provide a checkpoint path to evaluate!")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.get("load_first_bacth"):
        log.info("Loading first batch...")
        datamodule.prepare_data()
        datamodule.setup("test")
        dl = datamodule.test_dataloader(batch_size=2)
        example_input = next(iter(dl))
        model.example_input_array = example_input

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "logger": logger,
    #     "trainer": trainer,
    #     "ckpt_path": cfg.ckpt_path,
    # }

    # if logger:
    #     log.info("Logging hyperparameters!")
    #     utils.log_hyperparameters(object_dict)

    if cfg.get("test"):
        log.info("Starting testing!")
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

        try:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        except Exception as e:
            log.error(f"Error while testing: {e}")

    if cfg.get("evaluate"):
        log.info("Starting evaluation!")

        for key in cfg.eval:
            log.info(f"Instantiating evaluator {key}")
            evaluator = utils.instantiate_evaluator(
                cfg.eval[key],
                model_cfg=cfg.model,
                logger=logger,
                ckpt_path=cfg.ckpt_path,
                strict=cfg.strict,
            )

            try:
                log.info(f"Running evaluator {evaluator.__class__.__name__}")
                evaluator.run()
            except Exception as e:
                log.error(f"Error while running {evaluator}: {e}")

        # log.info("Instantiating evaluators ...")
        # evaluator_list: Optional[EvaluatorList] = utils.instantiate_evaluator_list(
        #     cfg.get("eval"),
        #     model_cfg=cfg.model,
        #     logger=logger,
        #     ckpt_path=cfg.ckpt_path,
        # )

        # if evaluator_list is not None:
        #     evaluator_list.run()

    # metric_dict = trainer.callback_metrics

    # return metric_dict, object_dict


# @hydra.main(config_path="../configs", config_name="eval.yaml", version_base=None)


@click.command()
@click.argument("ckpt_path", type=click.Path(exists=True))
@click.option("--eval_cfg", "-e", type=str, help="Evaluator config to run", default="evaluators")  # , multiple=True
@click.option("--devices", "-d", help="List of devices to use", multiple=True, type=int, default=None)
@click.option("--test", "-t", help="Test", default=False)
@click.option("--strict", "-s", help="Strict", default=False)
def main(ckpt_path: str, eval_cfg, devices, test, strict) -> None:
    """Main entrypoint for evaluation.

    Loads the config from the relative position of the checkpoint path.
    """
    # load config
    config_path = Path(ckpt_path).parent.parent / ".hydra/config.yaml"

    cfg = OmegaConf.load(config_path)

    cfg.paths.output_dir = Path(ckpt_path).parent.parent
    cfg.paths.work_dir = os.getcwd()

    cfg.ckpt_path = ckpt_path
    cfg.test = test
    if not test:
        cfg.load_first_bacth = False

    cfg.strict = strict
    cfg.task = "eval"

    cfg.logger.wandb.name += "_eval"

    # if isinstance(eval_cfg, str):
    #     eval_cfg = [eval_cfg]

    # eval_dict = {}

    # for eval_cfg_ in eval_cfg:
    #     eval_cfg_path = Path(cfg.paths.root_dir) / "configs" / "eval" / f"{eval_cfg_}.yaml"
    #     if not eval_cfg_path.exists():
    #         raise ValueError(f"Config for {eval_cfg_} not found!")

    #     abs_config_dir = str(eval_cfg_path.parent.parent.resolve())

    #     with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
    #         eval_cfg_ = compose(config_name=f"eval/{eval_cfg_}")

    #     eval_dict = OmegaConf.merge(eval_dict, eval_cfg_)

    eval_cfg_path = Path(cfg.paths.root_dir) / "configs" / "eval" / f"{eval_cfg}.yaml"
    if not eval_cfg_path.exists():
        raise ValueError(f"Config for {eval_cfg} not found!")

    abs_config_dir = str(eval_cfg_path.parent.parent.resolve())

    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        eval_cfg = compose(config_name=f"eval/{eval_cfg}")

    cfg.eval = eval_cfg.eval
    cfg.evaluate = True

    if devices is not None:
        cfg.trainer.devices = list(devices)

        for evaluator in cfg.eval:
            try:
                cfg.eval[evaluator].trainer.devices = list(devices)
            except AttributeError:
                print(evaluator)

    cfg.extras.print_eval = True

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
