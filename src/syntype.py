import sys
import logging
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts    import *
from utils      import *
from model      import *
from config     import *

logging.getLogger("vllm").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

printHeader(f"Classifying the Synonyms")

# To track time.
startTime = time.time()

# ------------------------------------------------------------------------------
# Load Human Phenotype Ontology (HPO) Data.
# ------------------------------------------------------------------------------

synonyms = None

# If already saved data is available.
if isFile(outputFileClass):
    synonyms = readCSV(outputFileClass)
else:
    # Only proceed if formatted input data exists
    exitIfFileNotExist(inputFileClass)

    # Load the dataset from a pickle file
    gold    = readCSV(inputFileClass)

    # Filter the synonyms.
    synonyms = gold[gold[classColumn].isin(synonymClasses)].copy().reset_index(
        drop = True)


    hpoIDs   = getHPOIDs(synonyms)
    parents  = {}
    children = {}

    # Collect parent's and children's labels for prompt creation. 
    with newProgress() as progress:
        
        task = newTask(progress, len(hpoIDs), "Get Parents and Children")

        for hpoID in hpoIDs:
            children[hpoID] = getChildLabels (gold, hpoID)
            parents[hpoID]  = getParentLabels(gold, hpoID)
            
            progress.update(task, advance = 1)
        
        progress.refresh()

    # Reduce the data to only neccessary.
    synonyms = synonyms[synonyms[hpoidColumn].isin(hpoIDs)].copy().reset_index(
        drop = True)

    with newProgress() as progress:

        task = newTask(progress, len(synonyms.index), "Creating first Prompt")
        messages = []

        for index, row in synonyms.iterrows():
            hpoID = row[hpoidColumn]

            # For the Chain-Of-Thoughts approach there are several other prompts
            # following after this. The Few-Shot approach is incorporated directly
            # into the prompts.
            if chainOfThoughts:
                messages.append(sourceClassificationPrompt1(
                    "".join(getElements(gold, hpoID, labelClass)),
                    "".join(getElements(gold, hpoID, definitionClass)),
                    "".join(getElements(gold, hpoID, commentClass)),
                    parents[hpoID],
                    children[hpoID]
                ))
            else:
                messages.append(sourceClassificationPrompt(
                    "".join(getElements(gold, hpoID, labelClass)),
                    "".join(getElements(gold, hpoID, definitionClass)),
                    "".join(getElements(gold, hpoID, commentClass)),
                    parents[hpoID],
                    children[hpoID],
                    row[contentColumn]
                ))

            progress.update(task, advance = 1)

        progress.refresh()

        synonyms["{}{}".format(userRole, 1)]  = messages
        synonyms["{}{}".format(modelRole, 1)] = [""] * len(synonyms.index)

        # Here the other prompts for the Chain-Of-Thoughts approach are added.
        # The Few-Shot approach is incorporated directl into the prompts.
        if chainOfThoughts:
            messages = []

            task = newTask(progress, len(synonyms.index), "Creating following Prompts")

            for index, row in synonyms.iterrows():
                hpoID = row[hpoidColumn]

                messages.append(sourceClassificationPrompt2(row[contentColumn]))
                progress.update(task, advance = 1)

            progress.refresh()

            synonyms["{}{}".format(userRole, 2)]  = messages
            synonyms["{}{}".format(modelRole, 2)] = [""] * len(synonyms.index)
            synonyms["{}{}".format(userRole, 3)]  = [sourceClassificationPrompt3()] * len(synonyms.index)
            synonyms[answerColumn]                = [""] * len(synonyms.index)

        synonyms[systemColumn] = [modelName] * len(synonyms.index)
        log("Write Prompts to file.")
        writeHugeCSV(synonyms, outputFileClass)

log(f"Set up the LLM ({modelName})...")
model = Model(model = modelID)
log(f"Set up of LLM complete.")

def structuredGeneration(data : pd.DataFrame = None, sourceColumn : str = "", 
    previousSourceColumnPrompts : list = [], previousSourceColumnRoles : list = [],
    targetColumn : str = "", file : str = "") -> pd.DataFrame:
    ret = None
    
    if data is not None:
        # Copy data.
        ret = data.copy()

        # Determining where to start generating. 
        answers = ret[targetColumn].tolist()

        
        if (any(pd.isna(x) or x == "" or x is None for x in answers)):
            
            startIndex = next(
                (i for i, x in enumerate(answers) if pd.isna(x) or x == "" or x is None), None
            )

            # Generate prompts
            for i in range(startIndex, len(ret.index), chunkSizeAnswerGeneration):

                # Reset model.
                model.reset()
                
                # Determine endIndex.
                endIndex = i + chunkSizeAnswerGeneration - 1
                if endIndex > len(ret.index):
                    endIndex = len(ret.index) - 1

                # Setup prompt history.
                for index in range(0, len(previousSourceColumnPrompts)):
                    p = ret.loc[i:endIndex, previousSourceColumnPrompts[index]].tolist()
                    model.addPrompt(previousSourceColumnRoles[index], p)

                # Add current Prompt
                p = ret.loc[i:endIndex, sourceColumn].tolist()
                c = model.addPrompt(userRole, p)

                log(f"{c} prompts added [Index: {i} - {endIndex}]. Start generating responses...")

                # Generate. 
                model.generate()

                # Process generated answer. 
                histories = model.getMessageHistories().copy()
                for index, history in enumerate(histories):
                    if (history is not None and isinstance(history, list) and history[-1][messageTextElement] is not None):
                        ret.loc[i + index, targetColumn] = str(history[-1][messageTextElement]).strip()

                # Save process. 
                writeHugeCSV(ret, file)
        else:
            log("Skipping Generation.")

    return ret



# ------------------------------------------------------------------------------
# Classification of synonyms.
# ------------------------------------------------------------------------------

log(f"(1) Generating.")
synonyms = structuredGeneration(synonyms, "{}{}".format(userRole, 1), [], [], "{}{}".format(modelRole, 1), outputFileClass)

# Here the other prompts for the Chain-Of-Thoughts approach are added.
# The Few-Shot approach is incorporated directl into the prompts.
if chainOfThoughts:
    log(f"(2) Generating.")
    synonyms = structuredGeneration(synonyms, "{}{}".format(userRole, 2), 
        ["{}{}".format(userRole, 1), "{}{}".format(modelRole, 1)], 
        [userRole, modelRole], 
        "{}{}".format(modelRole, 2), outputFileClass)

    log(f"(3) Generating.")
    synonyms = structuredGeneration(synonyms, "{}{}".format(userRole, 3), 
        ["{}{}".format(userRole, 1), "{}{}".format(modelRole, 1), "{}{}".format(userRole, 2), "{}{}".format(modelRole, 2)], 
        [userRole, modelRole, userRole, modelRole], 
        answerColumn, outputFileClass)
else:
    synonyms = synonyms.rename(columns={"{}{}".format(modelRole, 1): answerColumn})

# For time tracking.
minutes         = int((time.time() - startTime) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Synonyms Classified [Minutes: {minutes}]")