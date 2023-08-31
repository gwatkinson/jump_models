from pathlib import Path
from typing import List, Optional, Tuple

import click
import hydra
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src import utils
from src.eval import EvaluatorList

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


log = utils.get_pylogger(__name__)


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
        datamodule.setup("fit")
        dl = datamodule.train_dataloader(batch_size=2)
        example_input = next(iter(dl))
        model.example_input_array = example_input

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "ckpt_path": cfg.ckpt_path,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)  # ? This is enough to load the weights ?

    log.info("Starting evaluation!")
    log.info("Instantiating evaluators ...")
    evaluator_list: Optional[EvaluatorList] = utils.instantiate_evaluator_list(
        cfg.get("eval"),
        cross_modal_module=model,
        logger=logger,
        ckpt_path=cfg.ckpt_path,
    )

    if evaluator_list is not None:
        evaluator_list.run()

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


# @hydra.main(config_path="../configs", config_name="eval.yaml", version_base=None)


@click.command()
@click.argument("ckpt_path", type=click.Path(exists=True))
def main(ckpt_path: str) -> None:
    """Main entrypoint for evaluation.

    Loads the config from the relative position of the checkpoint path.
    """
    # load config
    config_path = Path(ckpt_path).parent.parent / ".hydra/config.yaml"

    cfg = OmegaConf.load(config_path)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
