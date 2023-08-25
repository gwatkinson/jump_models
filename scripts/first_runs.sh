#!/bin/bash

git pull

python src/train.py experiment=gin_context_pred/small trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=info_nce logger.wandb.project=first_real_runs logger.wandb.group=data_size
python src/train.py experiment=gin_context_pred/med trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=info_nce logger.wandb.project=first_real_runs logger.wandb.group=data_size
python src/train.py experiment=gin_context_pred/big trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=info_nce logger.wandb.project=first_real_runs logger.wandb.group=data_size

python src/train.py experiment=gin_context_pred/med trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=info_nce_reg logger.wandb.project=first_real_runs logger.wandb.group=info_nce
python src/train.py experiment=gin_context_pred/med trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=info_nce_temp logger.wandb.project=first_real_runs logger.wandb.group=info_nce
python src/train.py experiment=gin_context_pred/med trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=info_nce_reg model.criterion.temperature_requires_grad=True logger.wandb.project=first_real_runs logger.wandb.group=info_nce

python src/train.py experiment=gin_context_pred/med trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=ntxent logger.wandb.project=first_real_runs logger.wandb.group=ntxent
python src/train.py experiment=gin_context_pred/med trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=ntxent_reg logger.wandb.project=first_real_runs logger.wandb.group=ntxent
python src/train.py experiment=gin_context_pred/med trainer=ddp trainer.devices=[0,1] data.num_workers=32 seed=23540 model/criterion=ntxent_reg model.criterion.temperature_requires_grad=True logger.wandb.project=first_real_runs logger.wandb.group=ntxent
