"""Module to define our MAE pretrained model."""

# import gc
import pickle
from dataclasses import dataclass, field
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


def plot_example_pred(real, masked, pred, normalize=True, percentile=1.0, only_aggregate=False):
    nrow = 1 if only_aggregate else 6
    fig, axs = plt.subplots(nrow, 3, figsize=(12, 18))

    titles = ["Masked", "Reconstructed", "Real"]
    tensors = [masked, pred, real]

    for j in range(3):
        ex = tensors[j]

        if ex.dtype != np.uint8:
            ex = np.clip(ex * 255, 0, 255).astype(np.uint8)

        for i in range(6):
            if i == 0:
                if only_aggregate:
                    ax = axs[j]
                else:
                    ax = axs[i, j]
                ch1 = ex[0]
                ch2 = ex[1]  # (ex[1] + ex[3]) / 2
                ch3 = ex[2]  # (ex[2] + ex[4]) / 2

                if normalize:
                    ch1 = robust_convert_to_8bit(ch1, percentile=percentile)
                    ch2 = robust_convert_to_8bit(ch2, percentile=percentile)
                    ch3 = robust_convert_to_8bit(ch3, percentile=percentile)

                rgb = np.stack([ch2, ch3, ch1], axis=-1)

                ax.imshow(rgb)
                ax.axis("off")
                ax.set(title=titles[j])
            elif not only_aggregate:
                ax = axs[i, j]
                ch = ex[i - 1]
                kwargs = {}
                if normalize:
                    ch = robust_convert_to_8bit(ch, percentile=percentile)
                    kwargs = {"vmin": 0, "vmax": 255}

                ax.imshow(ch, cmap="gray", **kwargs)
                ax.axis("off")

    fig.tight_layout()

    return fig


def rxrx_to_img_paths(load_df, order=default_order):
    img_paths = []
    for _, row in load_df.iterrows():
        tmp = [row[f"w{i}"] for i in order]
        img_paths.append(tmp)

    return img_paths


def normalize_channel(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + np.finfo(float).eps)
    img = np.clip(img, 0, 1)

    img = (img * 255).astype(np.uint8)
    return img


def normalize_5_channel(img):
    new = np.stack([normalize_channel(x) for x in img], axis=0)
    return new


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

    def getitem(self, idx):
        image_path = self.image_paths[idx]

        image = load_image_paths_to_array(image_path)  # a 5*h*w array

        if image.shape == (5, 512, 512):
            image = scale_5_channel(image)

        if self.transform_fn:
            image = self.transform_fn(image)

        return image

    def __getitem__(self, idx):
        for t in range(10):
            try:
                return self.getitem(idx)
            except Exception as e:
                py_logger.error(f"Failed to get item {idx} with error: {e} (attempt {t})")
                idx = np.random.randint(0, len(self.image_paths))
        raise RuntimeError(f"Failed to get item {idx} after 10 attempts")

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
    train_test_val_split: List[float] = (0.98, 0.01, 0.01)
    batch_size: dict[str, int] = field(default_factory={"train": 32, "val": 32, "test": 32})
    num_workers: dict[str, int] = field(default_factory={"train": 12, "val": 12, "test": 12})
    prefetch_factor: int = 1
    pin_memory: bool = False
    drop_last: bool = True
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
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=True)

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
        self.transform = transform
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

        total_train_batch_size = self.batch_size["train"] * self.world_size
        self.lr = self.base_lr * total_train_batch_size / 256

        self.failed_once = False
        self.failed_once_normed = False

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
            "batch_size": self.batch_size[stage],
            "num_workers": self.num_workers[stage],
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
        idx = np.random.randint(0, len(batch))
        model = self.vit_mae_for_pretraining

        with torch.no_grad():
            real_image = batch[idx].detach().cpu().numpy()

            prediction = logits
            pred_image = model.unpatchify(prediction)[idx].detach().cpu().numpy()
            pred_image = np.clip(pred_image, 0, 1)

            if self.vit_config.norm_pix_loss:
                mean = np.mean(real_image, axis=(1, 2), keepdims=True)
                var = np.var(real_image, axis=(1, 2), keepdims=True)
                pred_image = pred_image * ((var + 1.0e-6) ** 0.5) + mean

            patches = model.patchify(batch)
            masks = mask[idx].unsqueeze(-1).expand_as(patches)
            masked_patches = torch.max(patches, masks) - masks * 0.9
            masked_images = model.unpatchify(masked_patches)[idx].detach().cpu().numpy()

            fig = plot_example_pred(real_image, masked_images, pred_image)

        del real_image, pred_image, patches, masks, masked_patches, masked_images

        return fig

    def model_step(self, batch, batch_idx, stage=None):
        res = self(batch)

        loss = res.loss
        # mean_loss = loss

        # losses = self.all_gather(loss)
        # mean_loss = torch.mean(losses)

        self.log(
            f"{stage}/loss",
            loss.item(),
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=(stage != "train"),
            sync_dist=True,
        )

        if (batch_idx % 250) == 0 and not self.failed_once:
            try:
                fig = self.plot_example_pred(batch, res.logits, res.mask)
                self.logger.log_image(f"{stage}/example_pred", [fig])
                plt.close(fig)
            except Exception as e:
                py_logger.warning(f"Could not plot example prediction: {e}")
                self.failed_once = True

        if not torch.isfinite(loss):
            py_logger.error(f"Loss of batch {batch_idx} is not finite: {loss}")
            return None

        # gc.collect()
        # torch.cuda.empty_cache()
        # gc.collect()

        return loss

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
