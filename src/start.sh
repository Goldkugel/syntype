#!/bin/bash

source "$(dirname "$0")/settings.sh"

clear

# Preparation: transforms the HPO and embeds its terms for the evaluation,
# each only if its output is missing or outdated. The jobs below classify the
# synonyms and format the answers.
python3 ./syntype.py "" || exit 1

# One job per experiment, in which all models of llm.model_id (config.yaml)
# classify one after the other.
sbatch classify.sh all "$MODE" "$CoT" "$FS" "" "" "" ""
sbatch classify.sh all "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
sbatch classify.sh all "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
sbatch classify.sh all "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
sbatch classify.sh all "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
