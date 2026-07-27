#!/bin/bash

MODE="test" 
#MODE=""

#CoT="chain-of-thoughts" 
CoT=""
#FS="" 
FS="few-shot"

DEFINITION="definition"
COMMENT="comment"
CHILDREN="children"
PARENTS="parents"

MODELS1=(
  "google/medgemma-4b-it"
  "Qwen/Qwen3-4B-Instruct-2507"
  "mistralai/Mistral-7B-Instruct-v0.2"
)

MODELS2=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "google/medgemma-27b-text-it"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)

clear

[ -f "../data/output/transform/transform.csv" ] || (python3 "transform.py" && python3 "embed.py" "" "" "test" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS" && python3 "embed.py" "" "" "" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS")

for MODEL in "${MODELS1[@]}"; do
  sbatch classify1.sh "$MODEL" "$MODE" "$CoT" "$FS" "" "" "" ""
  sbatch classify1.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
  sbatch classify1.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
  sbatch classify1.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
  sbatch classify1.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
done

for MODEL in "${MODELS2[@]}"; do
  sbatch classify2.sh "$MODEL" "$MODE" "$CoT" "$FS" "" "" "" ""
  sbatch classify2.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
  sbatch classify2.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
  sbatch classify2.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
  sbatch classify2.sh "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
done