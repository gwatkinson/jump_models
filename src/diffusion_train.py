import os
from pathlib import Path
from typing import List, Tuple

import click
import hydra
import pyrootutils
import wandb
from hydra import compose, initialize_config_dir
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from tqdm.auto import tqdm

from src import utils
from src.morpho2mol.models_mol2img_prior import DiffusionPrior, Mol2ImgDiffusionPrior, Mol2ImgDiffusionPriorNetwork
from src.morpho2mol.mol2img_prior_trainer import DiffusionPriorTrainer
from src.utils import color_log

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = color_log.get_pylogger(__name__)


def train_diffusion(mol_encoder, img_encoder, embed_dim, dataloader, epochs, device):
    img_encoder.eval()
    for p in img_encoder.parameters():
        p.requires_grad = False
    img_encoder.to(device)

    mol_encoder.eval()
    for p in mol_encoder.parameters():
        p.requires_grad = False
    mol_encoder.to(device)

    prior_network = Mol2ImgDiffusionPriorNetwork(dim=512, depth=6, dim_head=64, heads=8).to(device)

    diffusion_prior = Mol2ImgDiffusionPrior(
        net=prior_network,
        morpho_embed_dim=embed_dim,  # here the size of the ''clip'' embedding
        condition_on_mol_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        morpho_embed_scale=None,
    ).to(device)

    diffusion_prior.train()
    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior, lr=3e-4, wd=1e-2, ema_beta=0.99, ema_update_after_step=1000, ema_update_every=10
    )

    losses = []
    epoch_loss = 0
    for epoch in range(epochs):
        step_losses = []
        step_loss = 0
        dataloader_pbar = tqdm(dataloader)
        for img, mol in dataloader_pbar:
            dataloader_pbar.set_description(f"Epoch {epoch} | Epoch Loss {epoch_loss:.4f} | Step Loss {step_loss:.4f}")

            clip_img = img_encoder(img)
            clip_mol = mol_encoder(mol)
            loss = diffusion_prior_trainer(mol_embed=clip_mol, morpho_embed=clip_img, max_batch_size=4)
            diffusion_prior_trainer.update()

            step_loss = loss.item()
            step_losses.append(step_loss)

        epoch_loss = sum(step_losses) / len(step_losses)
        losses.append(epoch_loss)

    return losses


@utils.task_wrapper
def diffuse(cfg: DictConfig) -> Tuple[dict, dict]:
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    mol_encoder = nn.Sequential(model.molecule_encoder, model.molecule_projection_head)
    img_encoder = nn.Sequential(model.image_encoder, model.image_projection_head)
    embed_dim = model.image_projection_head.out_features

    dataloader = datamodule.train_dataloader(batch_size=4)

    device = cfg.device

    mol_encoder
    losses = train_diffusion(
        mol_encoder, img_encoder, embed_dim=embed_dim, dataloader=dataloader, epochs=100, device=device
    )

    return losses


@click.command()
@click.argument("ckpt_path", type=click.Path(exists=True))
@click.option("--eval_cfg", "-e", type=str, help="Evaluator config to run", default="evaluators")  # , multiple=True
@click.option("--devices", "-d", help="List of devices to use", multiple=True, type=int, default=None)
@click.option("--strict", "-s", help="Strict", default=True)
def main(ckpt_path: str, devices, strict) -> None:
    config_path = Path(ckpt_path).parent.parent / ".hydra/config.yaml"

    cfg = OmegaConf.load(config_path)

    cfg.paths.output_dir = Path(ckpt_path).parent.parent
    cfg.paths.work_dir = os.getcwd()

    cfg.ckpt_path = ckpt_path

    cfg.device = f"cuda:{devices[0]}" if devices is not None else "cpu"

    cfg.model["_target_"] += ".load_from_checkpoint"
    with open_dict(cfg.model):
        cfg.model["checkpoint_path"] = ckpt_path
        cfg.model["map_location"] = cfg.device
        cfg.model["strict"] = strict

    cfg.data["_target_"] = "src.models.jump_cl.datamodule.BasicJUMPDataModule"
    cfg.data["split_path"] = "train_big"

    diffuse(cfg)


if __name__ == "__main__":
    main()
