#!/bin/bash

git pull

python src/train.py \
    seed=2213 \
    experiment=coati/med \
    trainer=gpu \
    trainer.devices=[1] \
    trainer.max_epochs=200 \
    data.num_workers=32 \
    data.transform.size=224 \
    data.batch_size=128 \
    model.embedding_dim=256 \
    model/image_encoder=vit_base_16_224 \
    model/criterion=ntxent_reg \
    model.criterion.alpha=0.2 \
    model.criterion.mse_reg=0.5 \
    model.criterion.variance_reg=1 \
    model.criterion.covariance_reg=0.25 \
    logger.wandb.project=first_real_runs \
    logger.wandb.group=coati


gl && python src/train.py \
    seed=22123 \
    experiment=coati/med \
    trainer=gpu \
    trainer.devices=[1] \
    trainer.max_epochs=200 \
    data.num_workers=16 \
    data.transform.size=224 \
    data.batch_size=128 \
    model.embedding_dim=256 \
    model/image_encoder=vit_base_16_224 \
    model/criterion=ntxent_reg \
    model.criterion.alpha=0.2 \
    model.criterion.mse_reg=0.5 \
    model.criterion.variance_reg=1 \
    model.criterion.covariance_reg=0.25 \
    model.criterion.temperature=10 \
    model.criterion.temperature_requires_grad=True \
    logger.wandb.project=first_real_runs \
    logger.wandb.group=coati
