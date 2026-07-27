from owlready2 import get_ontology

import pandas as pd
import sys
import time

# Prevent Python from generating .pyc files.
sys.dont_write_bytecode = True

# Project-specific configuration and utility functions.
from config import *
from utils  import *

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

printHeader("Transforming the Raw HPO Data")

# To track time.
startTime = time.time()

# ------------------------------------------------------------------------------
# Load Human Phenotype Ontology (HPO) from OWL file.
# ------------------------------------------------------------------------------

# Ensure the input file exists before proceeding.
exitIfFileNotExist(inputFileTransformed)






# Load the ontology from the OWL file.
printRead(inputFileTransformed)
hpo = get_ontology(inputFileTransformed).load()
printReadDone(inputFileTransformed)

# ------------------------------------------------------------------------------
# Extract ontology content into tabular structures.
# ------------------------------------------------------------------------------

# This list will store multiple DataFrames, each representing a different
# type of extracted ontology information.
data = []

# --- Synonyms and their types ---
log("Retrieving Synonyms...")
data.append(getSynonymsAndTypes(hpo))
log(f"{len(data[-1].index)} Synonyms retrieved.")

# --- Comments associated with HPO terms ---
log("Retrieving Comments...")
data.append(getComments(hpo))
log(f"{len(data[-1].index)} Comments retrieved.")

# --- Definitions of HPO terms ---
log("Retrieving Definitions...")
data.append(getDefinitions(hpo))
log(f"{len(data[-1].index)} Definitions retrieved.")

# --- Preferred labels for HPO terms ---
log("Retrieving Labels...")
data.append(getLabels(hpo))
log(f"{len(data[-1].index)} Labels retrieved.")

# --- Child (subclass) relationships ---
log("Retrieving Children...")
data.append(getChildren(hpo))
log(f"{len(data[-1].index)} Children retrieved.")

# --- External references (e.g., publications, databases) ---
log("Retrieving References...")
data.append(getReferences(hpo))
log(f"{len(data[-1].index)} References retrieved.")

# ------------------------------------------------------------------------------
# Merge all extracted information into a single DataFrame.
# ------------------------------------------------------------------------------

log("Merging Data...")

# Combine all DataFrames, remove duplicates, and reset the index.
data = (
    pd.concat(data)
      .drop_duplicates(inplace = False, ignore_index = True)
      .reset_index(drop = True)
)

log(f"Merging resulted in {len(data.index)} Lines of Data.")

# ------------------------------------------------------------------------------
# Clean data: remove rows with empty or missing content.
# ------------------------------------------------------------------------------

rowCount = len(data.index)
log("Removing Empty Content Rows...")

# Remove rows where the content column is empty or NaN.
data = (
    data[data[contentColumn] != ""]
    .dropna(subset = [contentColumn, hpoidColumn])
    .reset_index(drop = True)
)

log(f"Removed {rowCount - len(data.index)} Rows due to Empty Content.")

log("Converting terms to lower case...")

with newProgress() as progress:
   
    task = newTask(progress, len(data.index), "Converting to lower case")
    c = ([labelClass] + synonymClasses)
    for index, row in data.iterrows():
        
        if row[classColumn] in c:
            data.loc[index, "tmp"] = str(data[contentColumn][index]).lower()
        progress.update(task, advance = 1)
    
    progress.refresh()

log("Terms were converted to lower case.")


rowCount = len(data.index)
log("Removing synonyms having a match with labels...")
data["tmp"] = data[hpoidColumn].astype(str) + data["tmp"].astype(str)
labeldata = data.loc[data[classColumn] == labelClass, "tmp"].tolist()
data = data[
    (~data[classColumn].isin(synonymClasses)) |
    (~data["tmp"].isin(labeldata))
].reset_index(drop = True).drop('tmp', axis = 1).copy()
log(f"Removed {rowCount - len(data.index)} synonyms due to having a " \
    "match with their label.")

printRowCount(data)
printDataSummary(data)

log("Write full transformated Data...")
writeCSV(data, outputFileTransformedFull)
log("Full transformed Data written.")

# ------------------------------------------------------------------------------
# Filter data to keep only selected HPO concepts.
# ------------------------------------------------------------------------------

if len(testIDs) > 0:
    log("Creating a reduced set of HPO Concepts...")

    rowCount = len(data.index)

    # If test IDs are provided, limit the data to:
    # - the test IDs themselves
    # - their children
    # - their parents
    if testIDs is not None and len(testIDs) > 0:
        parentIDs = data.loc[
            (data[contentColumn].isin(testIDs)) &
            (data[classColumn] == childrenClass),
            hpoidColumn
        ].tolist()

        childIDs = data.loc[
            (data[hpoidColumn].isin(testIDs)) &
            (data[classColumn] == childrenClass),
            contentColumn
        ].tolist()

        hpoIDs = list(set(testIDs + childIDs + parentIDs))

    # --------------------------------------------------------------------------
    # Reduce content to only the selected HPO concepts.
    # --------------------------------------------------------------------------

    result = data[data[hpoidColumn].isin(hpoIDs)].copy().reset_index(
        drop = True)
    log(f"Content Reduced by {rowCount - len(result.index)} Rows.")

    # --------------------------------------------------------------------------
    # Output summary statistics.
    # --------------------------------------------------------------------------

    printRowCount(result)
    printDataSummary(result)

    # --------------------------------------------------------------------------
    # Persist transformed data to disk.
    # --------------------------------------------------------------------------

    log("Write reduced transformated Data...")
    writeCSV(result, outputFileTransformed)
    log("Reduced transformed Data written.")






# For time tracking.
minutes         = int((time.time() - startTime) // 60)

# Printing Footer with minutes needed for the job.
printHeader(f"Transforming completed [Minutes: {minutes}]")