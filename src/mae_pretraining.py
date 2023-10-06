import logging
from functools import partial

import dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.mae.module import MAEDatasetConfig, MAEModule, MAEOptimizerConfig, ViTMAEConfig
from src.modules.transforms import SimpleTransform

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)


def main():
    transform = SimpleTransform(
        size=512,
        p=0.5,
    )

    optimizer_config = MAEOptimizerConfig(
        optimizer=partial(Lion, weight_decay=0.05, betas=(0.95, 0.9)),
        scheduler=partial(
            CosineAnnealingLR,
            T_max=50,
            eta_min=0.0,
            last_epoch=-1,
        ),
        lr=1e-4,
        monitor="val/loss",
        interval="epoch",
        frequency=1,
    )

    data_config = MAEDatasetConfig(
        train_test_val_split=(0.8, 0.1, 0.1),
        batch_size=128,
        transform=transform,
        prefetch_factor=2,
        pin_memory=True,
        num_workers=12,
        mae_dir="/projects/cpjump1/mae",
        use_jump=True,
        use_rxrx1=True,
        jump_load_df_path="/projects/cpjump1/jump/load_data/final",
        rxrx1_load_df_path="/projects/cpjump1/rxrx1/load_df.csv",
    )

    vit_config = ViTMAEConfig(
        image_size=512,
        patch_size=32,
        num_channels=5,
        mask_ratio=0.25,
        norm_pix_loss=True,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        decoder_num_attention_heads=8,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=2048,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )

    module = MAEModule(
        vit_config=vit_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
    )

    logger = [
        WandbLogger(project="mae", name="mae", log_model=True),
    ]

    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=2),
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            verbose=False,
            mode="min",
            check_finite=True,
        ),
        ModelCheckpoint(
            monitor="val/loss",
            save_last=True,
            save_top_k=1,
            mode="min",
        ),
        LearningRateMonitor(),
    ]

    trainer = Trainer(
        accelerator="gpu",
        strategy="auto",
        devices=[0, 1, 2],
        max_epochs=50,
        precision="16-mixed",
        sync_batchnorm=True,
        detect_anomaly=True,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
        # overfit_batches=3,
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()
