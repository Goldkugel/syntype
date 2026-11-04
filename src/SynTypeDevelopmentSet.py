"""
Draws the development set from the transformed HPO (transform.csv) and
writes it to hpo.reduced.csv: syntype.development_set.size_per_type expert
and layperson synonyms each, drawn at random with syntype.development_set.seed,
with the label, definition and comment of their concepts, the parent and
child relations of the concepts and the labels of those parents and
children, but no other synonym. Runs with "test" classify these synonyms.

The preparation run (python3 ./src/SynTypeClassification.py) draws the set
itself when it is missing or outdated; this script draws it again in any
case, e.g. after changing the seed while formulating the prompts.

    python3 ./src/SynTypeDevelopmentSet.py
"""

import sys
import shlex

# Prevent Python from generating .pyc files (compiled bytecode files).
sys.dont_write_bytecode = True

from SynTypeUtils import *
from config       import *

from logger       import Logger, lockedFile

# The logger takes the "logger" section of the configuration file; the
# command line of the run is its first line, since the log file is named
# after the start of the run only.
l = Logger(config = configFile)
l.log(f"Command: python3 {shlex.join(sys.argv)}")

l.printHeader("Drawing the Development Set")

# Under the lock of the preparation runs, which draw the set as well.
with lockedFile(outputFileTransformed):
    with l.step("Drawing the development set"):
        writeDevelopmentSet()
