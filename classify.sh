#!/bin/bash
#SBATCH --job-name=syntype
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# One experiment: every model of llm.model_id classifies the synonyms, one
# after the other ("all" as first argument), or only the model given as
# first argument. Two GPUs, which the larger models need; a model is spread
# over both.

python3 ./src/syntype.py "$@"
