<div align="center">

# Multimodal learning using the JUMP dataset

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0.4-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

## Description

This repository contains code used to launch Deep Learning experiments trained on the JUMP dataset.

### Idées random

- Minimiser la distance du Schrodinger Bridge pour apprendre conjointement les deux distributions

## :hammer_and_wrench: Installation

This project uses conda to create a virtual environment with some main dependencies (Python, CUDA, Poetry), then uses Poetry to install the rest of the dependencies via pip.

```bash
# clone project
git clone https://github.com/gwatkinson/jump_models
cd jump_models

# create conda environment and install dependencies
conda create -n jump_models -f conda-linux-64.lock          # for linux
# conda create -n jump_models -f conda-windows-64.lock      # for windows

# activate conda environment
conda activate jump_models

# install other dependencies and current project with poetry
poetry install

# install pre-commit hooks if you want
pre-commit install
# pre-commit run -a # run all hooks on all files
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
