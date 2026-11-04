#!/bin/bash

clear

# The answers are formatted by the classification jobs
# (SynTypeClassification.py). The preparation run brings the embeddings up to
# date, if the HPO was transformed again since.
python3 ./src/SynTypeClassification.py || exit 1

# Every experiment of syntype.experiments (config.yaml): its classifications
# of all models are merged and evaluated.
EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    python3 ./src/SynTypeMerge.py "$INDEX"
    python3 ./src/SynTypeEval.py "$INDEX"
done
