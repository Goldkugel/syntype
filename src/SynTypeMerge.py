"""
Merges the formatted classifications of all models of an experiment into one
file, which SynTypeEval.py evaluates.

    python3 ./src/SynTypeMerge.py <index>   (index in syntype.experiments)
"""

import pandas as pd
import sys
import shlex
import os

sys.dont_write_bytecode = True

from SynTypeUtils import *
from config     import *

from logger     import Logger

# The log file is named after the start of the run only, so the command line
# is its first line.
l             = Logger(config = configFile)
l.log(f"Command: python3 {shlex.join(sys.argv)}")

if experimentIndex is None:
    l.error("Give the index of the experiment in syntype.experiments, e.g. " \
        f"python3 {sys.argv[0]} 0.")
    sys.exit(1)

l.printHeader("Merging Generated Classifications")

with l.step("Merging the classifications"):
    # Only the models of llm.model_id; other files with the prefix of the
    # experiment (e.g. of a model no longer in the list) are left out.
    files   = {
        modelID : outputFileClassFormatted(modelNameOf(modelID))
        for modelID in configurationFile["llm"]["model_id"]
    }
    present = {modelID : file for modelID, file in files.items() if isFile(file)}
    missing = [modelID for modelID in files if modelID not in present]

    if len(missing) > 0:
        l.warning(f"No classification of {len(missing)} of {len(files)} model(s) " \
            f"(a job that failed or is still running?):")
        for modelID in missing:
            l.warning(f"- {modelID}")

    others = sorted(
        file for file in (os.listdir(inputFolderClassMerged) if os.path.isdir(inputFolderClassMerged) else [])
        if file.startswith(outputFileNameClassFormattedPrefix) and file.endswith(csvFileFormat)
            and os.path.join(inputFolderClassMerged, file) not in files.values()
    )
    if len(others) > 0:
        l.warning(f"Leaving out {len(others)} file(s) of models not in llm.model_id:")
        for file in others:
            l.warning(f"- {file}")

    if len(present) == 0:
        l.error(f"No classification of any model of llm.model_id found in " \
            f"{quote(inputFolderClassMerged)} for this experiment " \
            f"({quote(outputFileNameClassFormattedPrefix)}).")
        sys.exit(1)

    # All models should have classified the same synonyms; differences (e.g.
    # of another HPO version) are logged.
    dataframes = {modelID : readCSV(file) for modelID, file in present.items()}

    def synonymsOf(data : pd.DataFrame) -> set:
        """
        The (concept, synonym) pairs of a classification.
        """
        return set(keysOf(data))

    first, reference = next(iter(dataframes.items()))
    referenceSynonyms = synonymsOf(reference)
    for modelID, data in dataframes.items():
        modelSynonyms = synonymsOf(data)
        l.log(f"{quote(modelID)}: {len(data.index)} classified synonyms.")
        if modelSynonyms != referenceSynonyms:
            l.warning(f"{quote(modelID)} classified other synonyms than {quote(first)}: " \
                f"{len(modelSynonyms - referenceSynonyms)} only by {quote(modelID)}, " \
                f"{len(referenceSynonyms - modelSynonyms)} only by {quote(first)}.")

    merged = pd.concat(dataframes.values(), ignore_index = True)
    writeHugeCSV(merged, outputFileClassMerged)

    l.log(f"Merged the classifications of {len(dataframes)} model(s) ({len(merged.index)} rows).")
