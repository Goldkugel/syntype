#!/bin/bash
#SBATCH --job-name=syntype
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                         # Fordert 2x NVIDIA A40 an
#SBATCH --time=20-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# classify.sh <index> [<model ID>]: every model of llm.model_id (or only the
# given one) classifies the synonyms of the experiment, one after the other.
# The larger models need both GPUs.

# Sample with PyTorch: FlashInfer compiles its kernels with the system's nvcc,
# which is too old ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

python3 ./src/SynTypeClassification.py "$@"
