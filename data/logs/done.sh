#!/usr/bin/env bash

# Usage: ./done.sh [directory]
# Counts the finished classifications (one per model and experiment): the
# logger's "Step 'Classifying the synonyms (<model>)' completed in ..." line;
# a log file of a run with several models holds one line per finished model.

DIR="${1:-.}"
PATTERN="Step 'Classifying the synonyms \(.*\)' completed"

count=$(grep -rhE "$PATTERN" "$DIR" 2>/dev/null | wc -l)

echo "Finished classifications: $count"
