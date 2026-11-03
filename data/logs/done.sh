#!/usr/bin/env bash

# Usage: ./done.sh [directory]
# Counts the finished classifications (one per model and experiment); a log
# file of a run with several models holds one line per finished model.

DIR="${1:-.}"
SEARCH_STRING="Synonyms Classified"

count=$(grep -rhF "$SEARCH_STRING" "$DIR" 2>/dev/null | wc -l)

echo "Finished classifications (\"$SEARCH_STRING\"): $count"
