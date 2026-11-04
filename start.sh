#!/bin/bash

clear

# Preparation: transforms the HPO, draws the development set and embeds the
# development and the evaluation set, each only if outdated.
python3 ./src/SynTypeClassification.py || exit 1

# One classification job per experiment of syntype.experiments.
EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    sbatch classify.sh "$INDEX"
done
