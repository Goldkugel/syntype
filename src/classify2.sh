#!/bin/bash
#SBATCH --job-name=syntype
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: HH:MM:SS)
#SBATCH --output=../data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=../data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

MODEL="$1"
MODE="$2"
CoT="$3"
FS="$4"

DEFINITION="$5"
COMMENT="$6"
CHILDREN="$7"
PARENTS="$8"

python3 ./syntype.py "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"