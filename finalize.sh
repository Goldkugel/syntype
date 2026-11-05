#!/bin/bash

clear

# Brings the transformed HPO, the sets and the distances up to date.
python3 ./src/SynTypeClassification.py || exit 1

# Merges and evaluates the classifications of every experiment.
EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    # An experiment without any classification is not evaluated.
    python3 ./src/SynTypeMerge.py "$INDEX" && python3 ./src/SynTypeEval.py "$INDEX"
done
