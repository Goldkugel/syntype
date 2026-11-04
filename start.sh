#!/bin/bash

clear

# Preparation: transforms the HPO, draws the development set and embeds the
# development set and the HPO for the evaluation, each only if its output is
# missing or outdated. The jobs below classify the
# synonyms and format the answers.
python3 ./src/SynTypeClassification.py || exit 1

# One job per experiment of syntype.experiments (config.yaml), in which all
# models of llm.model_id classify one after the other.
EXPERIMENTS=$(python3 ./src/config.py) || exit 1
for ((INDEX = 0; INDEX < EXPERIMENTS; INDEX++)); do
    sbatch classify.sh "$INDEX"
done
