#!/bin/bash

git pull

python src/train.py evaluate=False experiment=fp/med trainer.devices=[1]

python src/train.py evaluate=False experiment=multi_loss/ntxent_grad trainer.devices=[2]

python src/train.py evaluate=False experiment=multi_loss/ntxent_vae trainer.devices=[2]

python src/train.py evaluate=False experiment=multi_loss/ntxent_vae_gim trainer.devices=[2]

