#!/bin/bash

git pull

# Med dataset
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-19_14-02-50/checkpoints/epoch_189.ckpt -d 1 -e evaluators -t

# Small dataset
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-25_14-48-12/checkpoints/epoch_071.ckpt -d 1 -e evaluators -t

# Large dataset
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-22_18-49-19/checkpoints/epoch_049.ckpt -d 1 -e evaluators -t
