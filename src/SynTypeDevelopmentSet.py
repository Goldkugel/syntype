"""
Draws the development set (hpo.development.csv) and the evaluation set
(hpo.evaluation.csv) from hpo.csv, see writeDevelopmentSet() in
SynTypeUtils.py. The preparation run draws them only when missing or
outdated; this script draws them in any case, e.g. after changing the seed.

    python3 ./src/SynTypeDevelopmentSet.py
"""

import sys
import shlex

sys.dont_write_bytecode = True

from SynTypeUtils import *
from config       import *

from logger       import Logger, lockedFile

# The log file is named after the start of the run only, so the command line
# is its first line.
l = Logger(config = configFile)
l.log(f"Command: python3 {shlex.join(sys.argv)}")

l.printHeader("Drawing the Development and Evaluation Set")

# Under the lock of the preparation runs, which draw the set as well.
with lockedFile(outputFileTransformedDevelopment):
    with l.step("Drawing the development and evaluation set"):
        writeDevelopmentSet()
