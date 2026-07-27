import sys
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from config     import *
from utils      import *

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

printHeader(f"Formatting Answers of Synonym's Classification")

# To track time.
startTime = time.time()

# ------------------------------------------------------------------------------
# Load Human Phenotype Ontology (HPO) data.
# ------------------------------------------------------------------------------

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassFormatted)

# Load the dataset from a pickle file
classified    = readCSV(inputFileClassFormatted)







classified[confidenceColumn] = [-1] * len(classified.index)

with newProgress() as progress:

    task = newTask(progress, len(classified.index), "Formatting Answers")
    for index in range(0, len(classified.index)):
        classified.loc[index, answerColumn], \
            classified.loc[index, confidenceColumn] = \
            formatAnswerClassification(str(classified[answerColumn][index]))
        progress.advance(task, advance = 1)

    progress.refresh()

# ------------------------------------------------------------------------------
# Persist transformed data to disk.
# ------------------------------------------------------------------------------

writeCSV(classified, outputFileClassFormatted)

log("Logging incorrect classified Synonyms...")

# For logging purposes the gold standard is read and wrong classifications
# are placed in the logging file. This is useful when it comes to prompt 
# optimization.
gold   = readCSV(inputFileClass)
labels = gold[gold[classColumn] == labelClass].copy().reset_index(drop = True)
count  = 0

for index, row in classified.iterrows():
    # It should log, when:
    # - The answer could not be formatted e.g. is undefined.
    # - If the answer or the gold class is "exact" or "related" but the other 
    #       is not equal.
    # This way all "exact" and "related" terms classified the wrong way are
    # being logged.
    #
    if (str(row[answerColumn]).lower() == undefinedSynonymType.lower() or
        (str(row[answerColumn]).lower() != str(row[classColumn]).lower()) and
        (str(row[answerColumn]).lower() in synonymClasses or
        str(row[classColumn]).lower() in synonymClasses)):
        log("Label: " \
            f"{applyFormat(getElements(labels, row[hpoidColumn], labelClass))}" \
            f", Synonym: {quote(row[contentColumn])}, " \
            f"Correct: {quote(row[classColumn])}, Classified: " \
            f"{quote(row[answerColumn])}", cmdline = False)
        count = count + 1

log(f"Incorrect Classifications: {count}")
log(f"Correct Classifications:   {len(classified.index) - count}")
log("Logging competed.")







# For time tracking.
minutes         = int((time.time() - startTime) // 60)

# Printing Footer with minutes needed for the job.
printHeader(f"Formatting completed [Minutes: {minutes}]")