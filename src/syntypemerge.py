import pandas as pd
import sys
import os
import time

# Prevent Python from generating .pyc files (compiled bytecode files).
sys.dont_write_bytecode = True

# Import project-specific helpers, model wrappers, and configuration.
from prompts    import *
from utils      import *

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

# Print a formatted header indicating the start of this processing stage.
printHeader(f"Merging Generated Classifications")

# To track time.
startTime = time.time()






# ------------------------------------------------------------------------------
# Load Data.
# ------------------------------------------------------------------------------

# Merging Generated Classes.
if len(inputFileClassMerged) == 0:
    log(f"No '{csvFileFormat}' files with prefix " + 
        f"'{outputFileNameClassFormattedPrefix}' found in " + 
        f"'{outputFolderNameFormatted}'")
else:
    # Read and merge CSV files.
    dataframes = []
    log("Reading files...")
    with newProgress() as progress:
        task = newTask(progress, len(inputFileClassMerged), 
            "Processing Files")

        for file in sorted(inputFileClassMerged):
            dataframes.append(pd.read_csv(file))
            progress.update(task, advance = 1)

        progress.refresh()

    # Read and merge CSV files.
    log("Files read.")

    log("Merging Data...")
    merged_df = pd.concat(dataframes, ignore_index = True)
    merged_df = merged_df.reset_index(drop = True)
    log("Data merged.")

    # --------------------------------------------------------------------------
    # Persist transformed data to disk.
    # --------------------------------------------------------------------------

    # Write merged CSV
    writeCSV(merged_df, outputFileClassMerged)

    log(f"Merged {len(inputFileClassMerged)} files into " + 
        f"'{os.path.basename(outputFileClassMerged)}'")






# For time tracking.
minutes         = int((time.time() - startTime) // 60)

# Print a formatted header indicating the end of this processing stage.
printHeader(f"Data Merged [Minutes: {minutes}]")