import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric

from src.modules.collate_fn import default_collate
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


@dataclass
class WSPCheckpointConfig:
    """Paths to load pretrained models from checkpoints.

    Not all of them are required.
    """

    module_ckpt: str = None  # Path to the module checkpoint (image and compound encoders, optimizer, etc.)
    image_backbone_ckpt: str = None  # Path to the image backbone checkpoint
    compound_backbone_ckpt: str = None  # Path to the compound backbone checkpoint
    image_mae_ckpt: str = None  # Path to the image masked autoencoder checkpoint
    graph_mae_ckpt: str = None  # Path to the graph masked autoencoder checkpoint


@dataclass
class WSPDatasetConfig:
    """Configuration for the dataset.

    One for train/val/test ?
    """

    return_images: bool = True  # Return transformed images
    return_graphs: bool = True  # Return molecular graphs
    return_batchs: bool = True  # Return batch id for each sample
    return_masked_graphs: bool = True  # Return masked graphs (atom and bond masking)
    return_context_graphs: bool = True  # Return context graphs
    image_transform: Any = None  # Image transformation
    compound_featurizer: Any = None  # Compound pre-transformation (from str to graph)
    graph_transforms: Any = None  # List of graph transformations (masking, etc.)
    data_root_dir: str = None  # Path to the data root directory (to replace in the load df for instance)
    image_sampler: Any = None  # Sampler function (to sample a or multiple images from a list of images)


@dataclass
class WSPDataloaderConfig:
    """DataLoader configuration."""

    batch_size: int = 32  # Batch size
    num_workers: int = 4  # Number of workers
    shuffle: bool = True  # Shuffle data
    drop_last: bool = True  # Drop last batch
    pin_memory: bool = False  # Pin memory
    prefetch_factor: int = 2  # Prefetch factor
    persistent_workers: bool = False  # Persistent workers
    collate_fn: Any = default_collate  # Collate function


@dataclass
class WSPDatamoduleConfig:
    """Configurations for the datamodule."""

    load_df_path: str = None  # Path to the load dataframe
    split_path: str = None  # Path to the split indexes
    train_ids_name: str = "train"  # Name of the train split file (val, test and retrieval are fixed)


@dataclass
class WSPOptimizerConfig:
    optimizer: Optional[Callable] = None
    scheduler: Optional[Callable] = None
    lr: float = 1e-4
    monitor: str = "val/loss"
    interval: str = "epoch"
    frequency: int = 1


@dataclass
class WSPLossConfig:
    use_info_nce: bool = True
    use_ntxent: bool = False
    use_simreg: bool = False
    use_mgm: bool = False
    use_mim: bool = False
    use_infocore: bool = False
    use_vrr: bool = False
    # coefs ?


@dataclass
class WSPModelConfig:
    # Usesless ?
    pass


@dataclass
class WSPLoggerConfig:
    # for plots, etc.
    pass


class WSPModule(LightningModule):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters(logger=True)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_jump(self) -> None:
        pass

    def prepare_data(self) -> None:
        pass

    def setup_load_df(self) -> None:
        pass
        # return load_df

    def setup_datasets(self) -> None:
        # ...
        # py_logger.info(
        #     f"Train, val, test splits: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}"
        # )
        pass

    def setup(self, stage: str = None, force=False) -> None:
        # if force or ...:
        pass

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

    def forward(self, x):
        pass

    def model_step(self, batch, batch_idx, mode="train"):
        pass

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, mode="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        pass
