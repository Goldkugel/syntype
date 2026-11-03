#!/bin/bash

source "$(dirname "$0")/settings.sh"

clear

# The answers are formatted by the classification jobs (syntype.py). The
# preparation run brings the embeddings up to date, if the HPO was
# transformed again since.
python3 ./src/syntype.py "" || exit 1

python3 ./src/syntypemerge.py "" "$MODE" "$CoT" "$FS" "" "" "" ""
python3 ./src/syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
python3 ./src/syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
python3 ./src/syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
python3 ./src/syntypemerge.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""

python3 ./src/syntypeeval.py "" "$MODE" "$CoT" "$FS" "" "" "" ""
python3 ./src/syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "" ""
python3 ./src/syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "" "$CHILDREN" "$PARENTS"
python3 ./src/syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "$CHILDREN" "$PARENTS"
python3 ./src/syntypeeval.py "" "$MODE" "$CoT" "$FS" "$DEFINITION" "$COMMENT" "" ""
