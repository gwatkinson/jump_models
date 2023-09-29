#!/bin/bash

git pull

# Med dataset
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-19_14-02-50/checkpoints/epoch_189.ckpt -d 1 -e retrieval -nt
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-19_14-02-50/checkpoints/epoch_189.ckpt -d 1 -e batch_effect -nt

# Small dataset
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-25_14-48-12/checkpoints/epoch_071.ckpt -d 1 -e evaluators -t

# Big dataset
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-22_18-49-19/checkpoints/epoch_049.ckpt -d 1 -e evaluators -t

# Med Multiview Intra
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-19_18-43-24/checkpoints/epoch_095.ckpt -d 1 -e evaluators -t

# Big Multiview Intra
python src/eval.py /import/pr_cpjump1/jump/logs/train/runs/2023-09-24_13-30-13/checkpoints/epoch_195.ckpt -d 1 -e evaluators -t

# Big Multiview Mol-Images
python src/eval.py /projects/cpjump1/jump/logs/train/runs/2023-09-26_10-03-49/checkpoints/last.ckpt -d 1 -e evaluators -t
