#!/bin/bash
# The experiment run by start.sh and evaluated by finalize.sh. Both read it
# from here, so they always work on the same files.

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
