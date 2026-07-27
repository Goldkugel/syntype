#!/bin/bash

#MODE="test" 
MODE=""

#CoT="chain-of-thoughts" 
CoT=""
#FS="" 
FS="few-shot"

DEFINITION="definition"
COMMENT="comment"
CHILDREN="children"
PARENTS="parents"

MODELS=(
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "google/medgemma-4b-it"
  "Qwen/Qwen3-4B-Instruct-2507"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "google/medgemma-27b-text-it"
)

clear

for MODEL in "${MODELS[@]}"; do
  python3 ./syntypeformat.py "$MODEL" "$MODE" "$CoT" "$FS" "" "" "" ""
  python3 ./syntypeformat.py "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
  python3 ./syntypeformat.py "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
  python3 ./syntypeformat.py "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
  python3 ./syntypeformat.py "$MODEL" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
done

python3 ./syntypemerge.py "" "$MODE" "$CoT" "$FS" "" "" "" ""
python3 ./syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
python3 ./syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
python3 ./syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
python3 ./syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""

python3 ./syntypeeval.py "" "$MODE" "$CoT" "$FS" "" "" "" ""
python3 ./syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
python3 ./syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
python3 ./syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
python3 ./syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
