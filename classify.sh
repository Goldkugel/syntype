#!/bin/bash
#SBATCH --job-name=syntype
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:2                        # Fordert 2x NVIDIA A40 an
#SBATCH --time=7-00:00:00                       # Maximale Laufzeit (Format: D-HH:MM:SS)
#SBATCH --output=./data/slurmlogs/job_%j.out        # Speicherort für Ausgaben (%j = Job-ID)
#SBATCH --error=./data/slurmlogs/job_%j.err         # Speicherort für Fehlermeldungen

# One experiment, given by its index in syntype.experiments (config.yaml):
# every model of llm.model_id classifies the synonyms, one after the other,
# or only the model given after the index (classify.sh <index> [<model ID>]).
# Two GPUs, which the larger models need; a model is spread over both.

# vLLM samples with PyTorch instead of FlashInfer: FlashInfer compiles its
# sampling kernels at startup with the system's nvcc, which is too old for
# its options ("nvcc fatal : Unknown option '--compress-mode=size'").
export VLLM_USE_FLASHINFER_SAMPLER=0

python3 ./src/SynTypeClassification.py "$@"
