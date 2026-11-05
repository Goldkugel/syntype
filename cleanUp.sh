#!/bin/bash

# Removes the outputs of the runs; the transformed HPO and the distances
# (with the embedding cache) take long to make and are kept.
rm -f ./data/logs/*.log
rm -f ./data/output/formatted/*.csv
rm -f ./data/output/evaluated/*.png ./data/output/evaluated/*.csv
rm -f ./data/output/merged/*.csv
rm -f ./data/output/transformed/*.csv ./data/output/transformed/*.json ./data/output/transformed/*.lock
rm -f ./data/output/embedded/*.csv ./data/output/embedded/*.json
rm -f ./data/output/type/*.csv
rm -f ./data/tmp/*.json ./data/tmp/*.lock
rm -f ./data/slurmlogs/*.err
rm -f ./data/slurmlogs/*.out

exit 0
