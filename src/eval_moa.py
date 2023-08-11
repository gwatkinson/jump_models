import logging

import torch
import torch.nn as nn
from dotenv import load_dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.callbacks import WandbPlottingCallback
from src.eval.moa.datamodule import JumpMOADataModule
from src.eval.moa.module import JumpMOAImageGraphModule, JumpMOAImageModule  # noqa: F401
from src.modules.compound_transforms import DGLPretrainedFromSmiles
from src.modules.images.timm_pretrained import CNNEncoder
from src.modules.molecules.dgllife_gin import GINPretrainedWithLinearHead  # noqa: F401
from src.modules.transforms import DefaultJUMPTransform
from src.splitters import ScaffoldSplitter, StratifiedSplitter  # noqa: F401

load_dotenv()


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

dataloader_config = DictConfig(
    {
        "train": {
            "batch_size": 128,
            "num_workers": 16,
            "shuffle": True,
        },
        "val": {
            "batch_size": 128,
            "num_workers": 16,
            "shuffle": False,
        },
        "test": {
            "batch_size": 128,
            "num_workers": 16,
            "shuffle": False,
        },
    }
)

logger.info("Loading DataModule")

dm = JumpMOADataModule(
    moa_load_df_path="/projects/cpjump1/jump/models/eval/test/moa_1000.csv",
    split_path="/projects/cpjump1/jump/models/eval/test/",
    dataloader_config=dataloader_config,
    force_split=False,
    transform=DefaultJUMPTransform(size=256),
    compound_transform=DGLPretrainedFromSmiles(),
    return_image=True,
    return_compound=True,
    collate_fn=None,
    metadata_dir="/projects/cpjump1/jump/metadata",
    load_data_dir="/projects/cpjump1/jump/load_data",
    splitter=StratifiedSplitter(
        train=0.75,
        val=0.15,
        test=0.1,
    ),
    max_obs_per_class=1000,
)


logger.info("Loading Models")
image_encoder = CNNEncoder("resnet18", target_num=256)
# molecule_encoder = GINPretrainedWithLinearHead("gin_supervised_infomax", out_dim=128)

logger.info("Setting up Module")
model = JumpMOAImageModule(
    image_encoder=image_encoder,
    optimizer=torch.optim.Adam,
    scheduler=None,
    criterion=nn.CrossEntropyLoss(),
    cross_modal_module=None,
    example_input_path="/projects/cpjump1/jump/models/eval/test/example.pt",
)

logger.info("Setting up Trainer")
callbacks = [
    WandbPlottingCallback(
        watch=True,
        watch_log="all",
        log_freq=50,
        log_graph=True,
        prefix="jump_moa/image_graph/",
        figsize=(16, 16),
        per_fmt=".1%",
        cmap="Blues_r",
        fz=10,
    )
]

loggers = WandbLogger(project="jump_moa", log_model=True, group="debug")
trainer = Trainer(
    max_epochs=50, logger=loggers, devices=[1], accelerator="gpu", log_every_n_steps=1, callbacks=callbacks
)

logger.info("Fitting Trainer")
trainer.fit(model, dm)
