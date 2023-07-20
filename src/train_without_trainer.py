import logging
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
import torch.nn as nn
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torchvision import disable_beta_transforms_warning
from tqdm.auto import tqdm

from src import utils

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


disable_beta_transforms_warning()
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info("Instantiating callbacks...")
    # callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # log.info("Instantiating loggers...")
    # logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")

        max_epochs: int = cfg.trainer.max_epochs
        device = torch.device("cuda" if cfg.trainer.accelerator == "gpu" else "cpu")

        model.train()

        optim_dict = model.configure_optimizers()
        optimizer = optim_dict["optimizer"]

        image_encoder: nn.Module = model.image_encoder.to(device)
        molecule_encoder: nn.Module = model.molecule_encoder.to(device)
        criterion: nn.Module = model.criterion.to(device)

        losses = []

        datamodule.setup("fit")

        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            optimizer.zero_grad()
            train_dl = datamodule.train_dataloader()
            epoch_loss = []

            for batch in tqdm(train_dl):
                c_emb = molecule_encoder(batch["compound"].to(device))
                i_emb = image_encoder(batch["image"].to(device))

                loss = criterion(c_emb, i_emb)

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

            mean_loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(mean_loss)

            pbar.set_description(f"Epoch {epoch} loss: {mean_loss}")
            pbar.refresh()


@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
