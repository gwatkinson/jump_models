#!/bin/bash

git pull

python src/train.py \
    experiment=gin_context_pred/big \
    seed=23540 \
    trainer=gpu \
    trainer.devices=[1] \
    trainer.max_epochs=200 \
    data.num_workers=32 \
    model/criterion=info_nce \
    logger.wandb.project=first_real_runs \
    logger.wandb.group=big_runs

python src/train.py \
    experiment=gin_context_pred/big \
    seed=23540 \
    trainer=gpu \
    trainer.devices=[1] \
    trainer.max_epochs=200 \
    data.num_workers=32 \
    model.lr=5e-4 \
    model/criterion=info_nce \
    logger.wandb.project=first_real_runs \
    logger.wandb.group=big_runs


python src/train.py \
    experiment=fp_big \
    seed=2131 \
    trainer=gpu \
    trainer.devices=[1] \
    trainer.max_epochs=100 \
    data.num_workers=32 \
    model/criterion=info_nce \
    logger.wandb.project=first_real_runs \
    logger.wandb.group=big_runs

python src/train.py \
    experiment=gin_context_pred/big \
    trainer=gpu trainer.devices=[1] trainer.max_epochs=100 \
    data.num_workers=32 \
    seed=213 \
    model/criterion=ntxent_reg \
    model.criterion.alpha=0.2 \
    model.criterion.mse_reg=0.5 \
    model.criterion.variance_reg=1 \
    model.criterion.covariance_reg=0.25 \
    logger.wandb.project=first_real_runs \
    logger.wandb.group=big_runs

