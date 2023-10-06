"""Module to define our MAE pretrained model."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers import ViTMAEConfig, ViTMAEForPreTraining

from src.utils import pylogger
from src.utils.io import load_image_paths_to_array

py_logger = pylogger.get_pylogger(__name__)


default_channels = ["DNA", "AGP", "ER", "Mito", "RNA"]
default_order = [1, 3, 2, 5, 4]


def jump_to_img_paths(load_df, channels=default_channels):
    img_paths = []
    for _, row in load_df.iterrows():
        tmp = [row[f"FileName_Orig{chan}"] for chan in channels]
        img_paths.append(tmp)

    return img_paths


def plot_example(ex):
    ex = ex.detach().cpu().numpy()

    if ex.dtype == np.float32:
        ex = (ex.numpy() * 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for i, ax in enumerate(axs.flatten()):
        if i < 5:
            ax.imshow(ex[i])
            ax.axis("off")
        elif i == 5:
            ch1 = ex[0]
            ch2 = ((ex[1] + ex[3]) / 2).astype(np.uint8)
            ch3 = ((ex[2] + ex[4]) / 2).astype(np.uint8)
            new = np.stack([ch2, ch1, ch3], axis=-1, dtype=np.uint8)
            ax.imshow(new)
            ax.axis("off")

    fig.tight_layout()

    return fig


def plot_example_pred(real, masked, pred):
    fig, axs = plt.subplots(6, 3, figsize=(12, 18))

    titles = ["Masked", "Reconstructed", "Real"]
    tensors = [masked, pred, real]

    for j in range(3):
        ex = tensors[j].detach().cpu().numpy()

        if ex.dtype != np.uint8:
            ex = (ex * 255).astype(np.uint8)

        for i in range(6):
            ax = axs[i, j]
            if i == 0:
                ch1 = ex[0]
                ch2 = ((ex[1] + ex[3]) / 2).astype(np.uint8)
                ch3 = ((ex[2] + ex[4]) / 2).astype(np.uint8)
                new = np.stack([ch2, ch1, ch3], axis=-1, dtype=np.uint8)
                ax.imshow(new)
                ax.axis("off")
                ax.set(title=titles[j])
            else:
                ax.imshow(ex[i - 1])
                ax.axis("off")

    fig.tight_layout()

    return fig


def rxrx_to_img_paths(load_df, order=default_order):
    img_paths = []
    for i, row in load_df.iterrows():
        tmp = [row[f"w{i}"] for i in order]
        img_paths.append(tmp)

    return img_paths


def robust_convert_to_8bit(img, percentile=1.0):
    """Convert a array to a 8-bit image by min percentile normalisation and
    clipping."""
    img = img.astype(np.float32)
    img = (img - np.percentile(img, percentile)) / (
        np.percentile(img, 100 - percentile) - np.percentile(img, percentile) + np.finfo(float).eps
    )
    img = np.clip(img, 0, 1)

    img = (img * 255).astype(np.uint8)
    return img


def scale_5_channel(img, percentile=1.0):
    new = np.stack([robust_convert_to_8bit(x, percentile) for x in img], axis=0)
    return new


class DiverseImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
    ):
        self.image_paths = image_paths  # list of (dataset, image_paths)
        self.transform_fn = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = load_image_paths_to_array(image_path)  # a 5*h*w array

        if image.shape == (5, 512, 512):
            image = scale_5_channel(image)

        if self.transform_fn:
            image = self.transform_fn(image)

        return image

    def plot_idx(self, idx):
        ex = self[idx]
        fig = plot_example(ex)
        return fig


@dataclass
class MAEDatasetConfig:
    mae_dir: str = "/projects/cpjump1/mae"
    use_jump: bool = True
    use_rxrx1: bool = True
    jump_load_df_path: str = "/projects/cpjump1/jump/load_data/final"
    rxrx1_load_df_path: str = "/projects/cpjump1/rxrx1/load_df.csv"
    transform: Optional[Callable] = None
    train_test_val_split: List[float] = (0.8, 0.1, 0.1)
    batch_size: int = 32
    prefetch_factor: int = 2
    pin_memory: bool = True
    num_workers: int = 4
    drop_last: bool = False
    persistent_workers: bool = False


@dataclass
class MAEOptimizerConfig:
    optimizer: Optional[Callable] = None
    scheduler: Optional[Callable] = None
    lr: float = 1e-4
    monitor: str = "val/loss"
    interval: str = "epoch"
    frequency: int = 1


class MAEModule(LightningModule):
    def __init__(
        self,
        vit_config: ViTMAEConfig,
        data_config: MAEDatasetConfig,
        optimizer_config: MAEOptimizerConfig,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # model setup
        self.vit_config = vit_config
        self.vit_mae_for_pretraining = ViTMAEForPreTraining(self.vit_config)

        # datasetup
        self.mae_dir = data_config.mae_dir
        self.use_jump = data_config.use_jump
        self.jump_load_df_path = data_config.jump_load_df_path
        self.use_rxrx1 = data_config.use_rxrx1
        self.rxrx1_load_df_path = data_config.rxrx1_load_df_path
        self.img_paths = None
        self.train_dataset = None
        self.transform = data_config.transform
        self.train_test_val_split = data_config.train_test_val_split
        self.batch_size = data_config.batch_size
        self.prefetch_factor = data_config.prefetch_factor
        self.pin_memory = data_config.pin_memory
        self.num_workers = data_config.num_workers
        self.drop_last = data_config.drop_last
        self.persistent_workers = data_config.persistent_workers

        # optimizer setup
        self.optimizer = optimizer_config.optimizer
        self.scheduler = optimizer_config.scheduler
        self.monitor = optimizer_config.monitor
        self.interval = optimizer_config.interval
        self.frequency = optimizer_config.frequency

        self.base_lr = optimizer_config.lr

        try:
            self.world_size = self.trainer.world_size
        except Exception:
            self.world_size = 1

        total_train_batch_size = self.batch_size * self.world_size
        self.lr = self.base_lr * total_train_batch_size / 256

        self.failed_once = False

    def prepare_jump(self) -> None:
        out_path = Path(self.mae_dir) / "jump.pickle"
        if out_path.exists():
            return

        py_logger.info(f"Loading jump load_df from {self.jump_load_df_path}")
        load_df = pd.read_parquet(self.jump_load_df_path)

        py_logger.info("Creating jump image paths")
        image_paths = jump_to_img_paths(load_df)
        py_logger.info(f"Number of images: {len(image_paths)}")

        py_logger.info(f"Saving jump image paths to {out_path}")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "wb") as f:
            pickle.dump(image_paths, f)

    def prepare_rxrx1(self) -> None:
        out_path = Path(self.mae_dir) / "rxrx1.pickle"
        if out_path.exists():
            return

        py_logger.info(f"Loading rxrx1 load_df from {self.rxrx1_load_df_path}")
        load_df = pd.read_csv(self.rxrx1_load_df_path)

        py_logger.info("Creating rxrx1 image paths")
        image_paths = rxrx_to_img_paths(load_df)
        py_logger.info(f"Number of images: {len(image_paths)}")

        py_logger.info(f"Saving rxrx1 image paths to {out_path}")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "wb") as f:
            pickle.dump(image_paths, f)

    def prepare_data(self) -> None:
        if self.use_rxrx1:
            self.prepare_rxrx1()
        if self.use_jump:
            self.prepare_jump()

    def setup_img_paths(self) -> None:
        imgs_paths = []

        if self.use_jump:
            in_path = Path(self.mae_dir) / "jump.pickle"

            py_logger.info(f"Loading jump image paths from {in_path}")
            with open(in_path, "rb") as f:
                jump_img_paths = pickle.load(f)

            imgs_paths.extend(jump_img_paths)
        if self.use_rxrx1:
            in_path = Path(self.mae_dir) / "rxrx1.pickle"

            py_logger.info(f"Loading rxrx1 image paths from {in_path}")
            with open(in_path, "rb") as f:
                rxrx_img_paths = pickle.load(f)

            imgs_paths.extend(rxrx_img_paths)

        return imgs_paths

    def setup_datasets(self) -> None:
        total_dataset = DiverseImageDataset(
            self.img_paths,
            transform=self.transform,
        )

        splits = torch.utils.data.random_split(
            total_dataset,
            lengths=self.train_test_val_split,
        )

        self.train_dataset = splits[0]
        self.val_dataset = splits[1]
        self.test_dataset = splits[2]

        py_logger.info(
            f"Train, val, test splits: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}"
        )

    def setup(self, stage: str = None, force=False) -> None:
        if self.img_paths is None or force:
            py_logger.info("Setting up image paths")
            self.img_paths = self.setup_img_paths()

        if self.train_dataset is None or force:
            py_logger.info("Setting up datasets")
            self.setup_datasets()

    def stage_dataloader(self, stage="train", **kwargs):
        dl_args = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "shuffle": (stage == "train"),
        }
        dl_args.update(kwargs)

        return DataLoader(getattr(self, f"{stage}_dataset"), **dl_args)

    def train_dataloader(self, **kwargs):
        return self.stage_dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.stage_dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.stage_dataloader("test", **kwargs)

    def predict_dataloader(self, **kwargs):
        pass

    def forward(self, x, **kwargs):
        # x shape: (B, 5, 512, 512)
        return self.vit_mae_for_pretraining(x, **kwargs)

    def plot_example_pred(self, batch, logits, mask):
        idx = 0
        model = self.vit_mae_for_pretraining

        with torch.no_grad():
            real_image = batch[idx].cpu()

            prediction = logits
            pred_image = model.unpatchify(prediction)[idx].detach().cpu()

            patches = model.patchify(batch)
            masks = mask.unsqueeze(-1).expand_as(patches)
            masked_patches = patches * masks
            masked_images = model.unpatchify(masked_patches)[idx].detach().cpu()

            fig = plot_example_pred(real_image, masked_images, pred_image)

        return fig

    def model_step(self, batch, batch_idx, stage=None):
        res = self(batch)

        loss = res.loss

        losses = self.all_gather(loss, sync_grads=True)
        mean_loss = torch.mean(losses)

        self.log(
            f"{stage}/loss",
            mean_loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"{stage}/pix_loss",
            loss,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        if (batch_idx % 250) == 0 and not self.failed_once:
            try:
                fig = self.plot_example_pred(batch, res.logits, res.mask)
                self.logger.experiment.log({f"{stage}/example_pred": fig})
            except Exception as e:
                print(f"Could not plot example prediction: {e}")
                self.failed_once = True

        if not torch.isfinite(mean_loss):
            py_logger.error(f"Loss of batch {batch_idx} is not finite: {mean_loss}")
            return None

        return mean_loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        if self.scheduler is not None:
            lr_scheduler_dict = {
                "scheduler": self.scheduler(optimizer=optimizer),
                "monitor": self.monitor,
                "interval": self.interval,
                "frequency": self.frequency,
                "strict": True,
                "name": "pretraining/lr",
            }

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_dict,
            }

        return {"optimizer": optimizer}
