#!/usr/bin/env bash

# Usage: ./count_synonyms.sh [directory]

DIR="${1:-.}"
SEARCH_STRING="Synonyms Classified"

count=$(grep -rlF "$SEARCH_STRING" "$DIR" 2>/dev/null | wc -l)

echo "Number of files containing \"$SEARCH_STRING\": $count"
