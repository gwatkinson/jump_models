#!/bin/bash

git pull

python src/train.py \
    experiment=gin_context_pred/big \
    trainer=gpu trainer.devices=[1] \
    data.num_workers=32 \
    seed=23540 \
    model/criterion=info_nce \
    logger.wandb.project=big_runs logger.wandb.group=gin_context_pred

python src/train.py \
    experiment=fp_big \
    trainer=gpu trainer.devices=[1] \
    data.num_workers=32 \
    seed=2131 \
    model/criterion=info_nce \
    logger.wandb.project=big_runs logger.wandb.group=fingerprints

python src/train.py \
    experiment=gin_context_pred/big \
    trainer=gpu trainer.devices=[1] \
    data.num_workers=32 \
    seed=23540 \
    model/criterion=ntxent_reg model.criterion.alpha=0.1 \
    logger.wandb.project=big_runs logger.wandb.group=gin_context_pred

