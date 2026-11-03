import pandas as pd
import sys
import shlex
import os
import time

# Prevent Python from generating .pyc files (compiled bytecode files).
sys.dont_write_bytecode = True

# Import project-specific helpers, model wrappers, and configuration.
from utils      import *
from config     import *

from logger     import Logger

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

# The logger takes the "logger" section of the configuration file; the
# command line of the run is its first line, since the log file is named
# after the start of the run only.
l             = Logger(config = configFile)
l.log(f"Command: python3 {shlex.join(sys.argv)}")

# Print a formatted header indicating the start of this processing stage.
l.printHeader(f"Merging Generated Classifications")

# To track time.
startTime = time.time()






# ------------------------------------------------------------------------------
# Load Data.
# ------------------------------------------------------------------------------

# The formatted classifications of all models of this experiment.
inputFileClassMerged = []
if os.path.isdir(inputFolderClassMerged):
    inputFileClassMerged = [
        os.path.join(inputFolderClassMerged, file)
        for file in os.listdir(inputFolderClassMerged)
        if file.startswith(outputFileNameClassFormattedPrefix) and
            file.endswith(csvFileFormat)
    ]

# Merging Generated Classes.
if len(inputFileClassMerged) == 0:
    l.log(f"No '{csvFileFormat}' files with prefix " + 
        f"'{outputFileNameClassFormattedPrefix}' found in " + 
        f"'{outputFolderNameFormatted}'")
else:
    # Read and merge CSV files.
    dataframes = []
    l.log("Reading files...")
    with newProgress() as progress:
        task = newTask(progress, len(inputFileClassMerged), 
            "Processing Files")

        for file in sorted(inputFileClassMerged):
            dataframes.append(pd.read_csv(file))
            progress.update(task, advance = 1)

        progress.refresh()

    # Read and merge CSV files.
    l.log("Files read.")

    l.log("Merging Data...")
    merged_df = pd.concat(dataframes, ignore_index = True)
    merged_df = merged_df.reset_index(drop = True)
    l.log("Data merged.")

    # --------------------------------------------------------------------------
    # Persist transformed data to disk.
    # --------------------------------------------------------------------------

    # Write merged CSV
    writeCSV(merged_df, outputFileClassMerged)

    l.log(f"Merged {len(inputFileClassMerged)} files into " + 
        f"'{os.path.basename(outputFileClassMerged)}'")






# For time tracking.
minutes         = int((time.time() - startTime) // 60)

# Print a formatted header indicating the end of this processing stage.
l.printHeader(f"Data Merged [Minutes: {minutes}]")