import os
import sys
import copy
import shlex
import fcntl
import json
import logging
import math
import time

import numpy  as np
import pandas as pd

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from utils      import *
from config     import *

from logger     import Logger

# The steps of syntype. The packages of a step (adapter, embedder, model)
# are imported by the step itself, so that a run only loads what it needs:
# the preparation run (no model given) does not load vLLM, the
# classification runs (start.sh) do not load the embedding models.
#
#   python3 syntype.py ""                  Preparation: transforms the HPO
#                                          (if needed) and embeds its terms
#                                          (if needed) for the evaluation.
#   python3 syntype.py all <words...>      Classification: transforms the HPO
#                                          (if needed), then every model of
#                                          llm.model_id classifies the
#                                          synonyms and its answers are
#                                          formatted, one model after the
#                                          other.
#   python3 syntype.py <model> <words...>  The same with only this model.

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

# The logger takes the "logger" section of the configuration file; the
# command line of the run is its first line, since the log file is named
# after the start of the run only.
l = Logger()
l.log(f"Command: python3 {shlex.join(sys.argv)}")

# The packages (adapter, embedder, model) take their sections of the
# configuration file; syntype.py sets the model and the files of a run on
# this copy of it.
configuration = copy.deepcopy(configurationFile)

def isOutdated(output : str, inputs : list) -> bool:
    """
    Whether a file is missing or older than one of the files it is made from.
    """
    return not isFile(output) or any(
        os.path.getmtime(output) < os.path.getmtime(i) for i in inputs if isFile(i))

# ------------------------------------------------------------------------------
# The settings an output was made with are stored next to it
# (<output>.settings.json), so that it is made again when they change, not
# only when one of its input files is newer.
# ------------------------------------------------------------------------------

# The transformation depends on the adapter settings and the test set.
transformSettings = {
    "adapter"  : configurationFile["adapter"],
    "test_ids" : testIDs
}

# The embedding depends on every embedder setting except those that do not
# change the vectors, and on the names and metrics of the similarity columns.
embedderSettingsWithoutEffect = ["device", "backend", "num_threads",
    "batch_size", "cache_folder", "cache_decimals", "progress_interval"]
embedSettings = {
    "embedder"              : {
        key : value for key, value in configurationFile["embedder"].items()
        if key not in embedderSettingsWithoutEffect
    },
    "embedding_model_names" : syntypeConfig.embedding_model_names,
    "similarity_metrics"    : similarityMetrics
}

def settingsFile(output : str) -> str:
    return output + ".settings.json"

def changedSettings(output : str, settings : dict) -> list:
    """
    The settings (top-level keys) that differ from those the output was made
    with; all of them if they were not recorded.
    """
    try:
        with open(settingsFile(output), "r", encoding = "utf-8") as f:
            recorded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return list(settings.keys())

    # Through JSON, as the recorded settings were, e.g. tuples become lists.
    current = json.loads(json.dumps(settings))
    return [key for key in current if recorded.get(key) != current[key]]

def writeSettings(output : str, settings : dict) -> None:
    with open(settingsFile(output), "w", encoding = "utf-8") as f:
        json.dump(settings, f, indent = 2)

def outdated(output : str, inputs : list, settings : dict, log : bool = True) -> bool:
    """
    Whether an output is missing, older than one of its inputs, or made with
    other settings; logs why, unless log is False.
    """
    if isOutdated(output, inputs):
        if log: l.log(f"{os.path.basename(output)} is missing or older than " \
            f"{applyFormat([os.path.basename(i) for i in inputs])}.")
        return True

    changed = changedSettings(output, settings)
    if len(changed) > 0:
        if log: l.log(f"{os.path.basename(output)} was made with other or unrecorded " \
            f"settings: {applyFormat(changed)}.")
        return True

    return False

def transformOutdated(log : bool = True) -> bool:
    outputs = [outputFileTransformedFull] + ([outputFileTransformed] if len(testIDs) > 0 else [])
    return any(outdated(output, [inputFileTransformed], transformSettings, log) for output in outputs)

# ==============================================================================
# Prompts: the templates of the experiment (prompt/) with the concept
# information, the synonym and the Few-Shot examples (fewShotExamples in
# config.py).
# ==============================================================================

def createJSONExample(cl : str = "", co : str = "10") -> str:
    ret = "{\n" \
        f"\t{quote(synonymClass)}: {quote(cl)},\n" \
        f"\t{quote(confidenceColumn)}: {co}\n" \
        "}"
    
    return ret

def formatInput(label : str = "", definition : str = "", comment : str = "", parents : list = [], children : list = [], synonym : str = "") -> str:
    ret = f"Label: {quote(label)}"

    if addDefinition:
        ret = ret + f"\nDefinition: {quote(definition)}"
    if addComment:
        ret = ret + f"\nComment: {quote(comment)}"
    if addParents:
        ret = ret + f"\nParents: {applyFormat(parents)}"
    if addChildren:
        ret = ret + f"\nChildren: {applyFormat(children)}"
    if len(synonym) > 0:
        ret = ret + f"\nSynonym: {quote(synonym)}"

    return ret

def getExamples() -> str:
    ret = "Examples"

    for index, example in enumerate(fewShotExamples):
        ret += f"""

Example {index + 1}

Input:
{formatInput(example["label"], example["definition"], example["comment"], example["parents"], example["children"], example["synonym"])}

Output:
{createJSONExample(example["type"], example["confidence"])}"""

    return ret

# The templates are filled in like Model.formatPromptFromFile() does.

def readTemplate(file : str) -> str:
    with open(os.path.join(promptFolder, file), "r", encoding = "utf-8") as f:
        return f.read()

templates = [readTemplate(file) for file in promptFiles]

def examples(fewShot : bool = fewShot) -> str:
    """
    The value of {examples}: the Few-Shot examples, or nothing.
    """
    return "\n\n" + getExamples() if fewShot else ""

def sourceClassificationPrompt1(
        label : str, 
        definition : str, 
        comment : str,  
        parents : list,
        children : list
) -> str:
    """
    First prompt of the Chain-Of-Thoughts approach: the HPO concept.
    """
    return templates[0].format(
        input = formatInput(label, definition, comment, parents, children, ""))

def sourceClassificationPrompt2(
    synonym : str
) -> str:
    """
    Second prompt of the Chain-Of-Thoughts approach: the synonym.
    """
    return templates[1].format(synonym = synonym)

def sourceClassificationPrompt3(fewShot : bool = fewShot) -> str:
    """
    Last prompt of the Chain-Of-Thoughts approach: the classification.
    """
    return templates[2].format(examples = examples(fewShot))

def sourceClassificationPrompt(
        label : str, 
        definition : str, 
        comment : str, 
        parents : list, 
        children : list,
        synonym : str,
        fewShot : bool = fewShot
)-> str:
    """
    The prompt of the single prompt approach.
    """
    return templates[0].format(
        input    = formatInput(label, definition, comment, parents, children, synonym),
        examples = examples(fewShot))

# ==============================================================================
# Transformation: hp.owl (adapter) -> transform.csv, transform.reduced.csv.
# ==============================================================================

def transformHPO() -> None:
    from adapter    import HPOAdapter, semanticClass, sourceType
    from adapter    import synonymClass as adapterSynonymClass

    # A synonym of a concept with all classes or all types of one of these sets
    # has no unambiguous gold label and is removed.
    conflictingSynonymClasses = {exactSynonymClass, relatedSynonymClass}
    conflictingSynonymTypes   = {laypersonSynonymType, abbreviationSynonymType}

    class SyntypeHPOAdapter(HPOAdapter):
        """
        HPOAdapter that removes the synonyms listed for the same concept both as
        exact and related synonym, or both as layperson synonym and
        abbreviation (an expert term).

        This is done in _clean(), which runs before the adapter removes
        duplicate (id, attribute, value) rows: afterwards only the first class
        and type of such a synonym would be left.
        """

        def _clean(self, data : pd.DataFrame) -> pd.DataFrame:
            data = super()._clean(data)

            synonyms   = data[data[self.config.attribute_column] == adapterSynonymClass]
            additional = synonyms[self.config.additional_column]

            # One key per synonym of a concept.
            key = (synonyms[self.config.id_column].astype(str) + "\t" +
                synonyms[self.config.value_column].astype(str))

            classes = additional.map(lambda a: a.get(semanticClass, "")).groupby(key).agg(set)
            types   = additional.map(lambda a: a.get(sourceType, "")).groupby(key).agg(set)

            conflictingClasses = classes.map(lambda c: conflictingSynonymClasses <= c)
            conflictingTypes   = types.map(lambda t: conflictingSynonymTypes <= t)
            l.log(f"Removing {conflictingClasses.sum()} synonyms listed as both " \
                f"{applyFormat(sorted(conflictingSynonymClasses))} and " \
                f"{(conflictingTypes & ~conflictingClasses).sum()} further synonyms " \
                f"listed as both {applyFormat(sorted(conflictingSynonymTypes))}...")

            remove = key.map(conflictingClasses | conflictingTypes)
            return data.drop(synonyms.index[remove.to_numpy()])

    l.printHeader("Transforming the Raw HPO Data")

    # To track time.
    startTime = time.time()

    # Ensure the input file exists before proceeding.
    exitIfFileNotExist(inputFileTransformed)

    # --------------------------------------------------------------------------
    # Load the HPO with the adapter. It extracts labels, definitions, comments,
    # children, references and synonyms of the active (not obsolete) concepts,
    # and removes rows without an ID or content and duplicates.
    # --------------------------------------------------------------------------

    adapter = SyntypeHPOAdapter(configuration)
    adapter.load()

    # --------------------------------------------------------------------------
    # Convert the adapter's rows (id, attribute, value, additional) into the
    # columns used by syntype.
    # --------------------------------------------------------------------------

    l.log("Converting Data...")

    hpo = adapter.data
    synonymRows = hpo[adapter.config.attribute_column] == adapterSynonymClass
    additional  = hpo[adapter.config.additional_column]

    # Synonyms get their synonym class (exact, related, ...) as class and their
    # source type (layperson, abbreviation, ...) as type; synonyms without a
    # source type (direct) are expert synonyms. All other rows keep their
    # attribute (label, definition, ...) as class.
    data = pd.DataFrame({
        hpoidColumn   : hpo[adapter.config.id_column],
        classColumn   : hpo[adapter.config.attribute_column].where(
            ~synonymRows, additional.map(lambda a: a.get(semanticClass, ""))),
        typeColumn    : additional.map(lambda a: a.get(sourceType, "")).where(
            synonymRows, "").replace(directSynonymType, expertSynonymType),
        contentColumn : hpo[adapter.config.value_column].astype(str),
        systemColumn  : goldStandardSystem
    }).reset_index(drop = True)

    l.log(f"Converting resulted in {len(data.index)} Lines of Data.")

    # --------------------------------------------------------------------------
    # Clean data: remove rows with empty or missing content.
    # --------------------------------------------------------------------------

    rowCount = len(data.index)
    l.log("Removing Empty Content Rows...")

    # Remove rows where the content column is empty or NaN.
    data = (
        data[data[contentColumn] != ""]
        .dropna(subset = [contentColumn, hpoidColumn])
        .reset_index(drop = True)
    )

    l.log(f"Removed {rowCount - len(data.index)} Rows due to Empty Content.")

    l.log("Converting terms to lower case...")

    # Labels and synonyms in lower case, to find synonyms matching their label.
    data["tmp"] = data[contentColumn].astype(str).str.lower().where(
        data[classColumn].isin([labelClass] + synonymClasses))

    l.log("Terms were converted to lower case.")


    rowCount = len(data.index)
    l.log("Removing synonyms having a match with labels...")
    data["tmp"] = data[hpoidColumn].astype(str) + data["tmp"].astype(str)
    labeldata = data.loc[data[classColumn] == labelClass, "tmp"].tolist()
    data = data[
        (~data[classColumn].isin(synonymClasses)) |
        (~data["tmp"].isin(labeldata))
    ].reset_index(drop = True).drop('tmp', axis = 1).copy()
    l.log(f"Removed {rowCount - len(data.index)} synonyms due to having a " \
        "match with their label.")

    l.printRowCount(data)
    printDataSummary(data)

    l.log("Write full transformated Data...")
    writeCSV(data, outputFileTransformedFull)
    writeSettings(outputFileTransformedFull, transformSettings)
    l.log("Full transformed Data written.")

    # --------------------------------------------------------------------------
    # Filter data to keep only selected HPO concepts.
    # --------------------------------------------------------------------------

    if len(testIDs) > 0:
        l.log("Creating a reduced set of HPO Concepts...")

        rowCount = len(data.index)

        # Limit the data to:
        # - the test IDs themselves
        # - their children
        # - their parents
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

        # ----------------------------------------------------------------------
        # Reduce content to only the selected HPO concepts.
        # ----------------------------------------------------------------------

        result = data[data[hpoidColumn].isin(hpoIDs)].copy().reset_index(
            drop = True)
        l.log(f"Content Reduced by {rowCount - len(result.index)} Rows.")

        # ----------------------------------------------------------------------
        # Output summary statistics.
        # ----------------------------------------------------------------------

        l.printRowCount(result)
        printDataSummary(result)

        # ----------------------------------------------------------------------
        # Persist transformed data to disk.
        # ----------------------------------------------------------------------

        l.log("Write reduced transformated Data...")
        writeCSV(result, outputFileTransformed)
        writeSettings(outputFileTransformed, transformSettings)
        l.log("Reduced transformed Data written.")

    # For time tracking.
    minutes         = int((time.time() - startTime) // 60)

    # Printing Footer with minutes needed for the job.
    l.printHeader(f"Transforming completed [Minutes: {minutes}]")

# ==============================================================================
# Embedding: the similarity of every synonym to its label -> embed.csv, for
# the evaluation (syntypeeval.py).
# ==============================================================================

def embedTerms() -> None:
    from embedder import Embedder

    # Printing header.
    l.printHeader("Embedding HPO Terms")

    # To track time.
    startTime = time.time()

    # --------------------------------------------------------------------------
    # Load the Human Phenotype Ontology (HPO) from the transformed data file.
    # All concepts are embedded, also for test runs: the similarities are
    # standardised over all synonyms, so that test and full runs share one scale.
    # --------------------------------------------------------------------------

    # Ensure the input file exists before proceeding.
    exitIfFileNotExist(outputFileTransformedFull)

    data = readCSV(outputFileTransformedFull)

    l.log("Filtering Labels and Synonyms of HPO Concepts...")
    # One label per concept (the first, if a concept has several).
    labelRows   = data[data[classColumn] == labelClass].drop_duplicates(
        subset = hpoidColumn)
    labels      = labelRows[contentColumn].astype(str).tolist()

    synonymRows = data[data[classColumn].isin(synonymClasses)][
        [hpoidColumn, classColumn, contentColumn]].reset_index(drop = True)
    synonyms    = synonymRows[contentColumn].astype(str).tolist()

    # Position of the label of each synonym's concept in labels (-1 if the
    # concept has no label).
    labelIndex  = synonymRows[hpoidColumn].map(
        pd.Series(range(len(labels)), index = labelRows[hpoidColumn])
    ).fillna(-1).astype(int).to_numpy()
    hasLabel    = labelIndex >= 0
    l.log(f"Found {len(labels)} labels and {len(synonyms)} synonyms " \
        f"({(~hasLabel).sum()} of a concept without label).")

    def similarity(metric : str, a : np.ndarray, b : np.ndarray) -> np.ndarray:
        """
        Similarity of the rows of a and b with the metric (larger is more
        similar), or None for an unknown metric.
        """
        ret = None

        if metric == cosineSimilarity or metric == angularSimilarity:
            ret = np.sum(a * b, axis = 1) / (
                np.linalg.norm(a, axis = 1) * np.linalg.norm(b, axis = 1))
            if metric == angularSimilarity:
                ret = 1 - np.arccos(np.clip(ret, -1.0, 1.0)) / np.pi
        elif metric == euclideanSimilarity or metric == mahalanobisSimilarity:
            # Without a covariance matrix the Mahalanobis distance is the
            # Euclidean distance.
            ret = 1 / (1 + np.linalg.norm(a - b, axis = 1))
        elif metric == manhattanSimilarity:
            ret = 1 / (1 + np.sum(np.abs(a - b), axis = 1))

        return ret

    # --------------------------------------------------------------------------
    # Embed and calculate the similarity of labels with their respective synonyms.
    # --------------------------------------------------------------------------

    l.log("Start to Embed Synonyms and Labels with all Embedding Models.")
    embeddingModelsList = sorted(embeddingModels.keys())
    for index, embeddingModel in enumerate(embeddingModelsList):

        # Set up the embedding model. One model is loaded at a time, so only one
        # is held in memory.
        l.log(f"Set up ({index + 1}/{len(embeddingModelsList)}) " \
            f"model {quote(embeddingModel)}...")
        configuration["embedder"]["model_id"] = [embeddingModels[embeddingModel]]
        embedder = Embedder(configuration)
        l.log("Model set up.")

        l.log("Embedding Labels...")
        labelEmbeddings   = np.asarray(
            embedder.embed(labels)[embeddingModels[embeddingModel]])
        l.log("Embedding Synonyms...")
        synonymEmbeddings = np.asarray(
            embedder.embed(synonyms)[embeddingModels[embeddingModel]])
        l.log("Labels and Synonyms embedded.")

        # The label embedding of every synonym's concept.
        a = synonymEmbeddings[hasLabel]
        b = labelEmbeddings[labelIndex[hasLabel]]

        m = "mean"
        s = "std"

        for similarityMetric in similarityMetrics:
            similarityColumn = similarityColumnPrefix.format(embeddingModel,
                similarityMetric)

            # Standard value for no similarity is np.nan.
            values           = np.full(len(synonyms), np.nan)
            values[hasLabel] = similarity(similarityMetric, a, b)

            # Normalization of the similarity metrics: since every metric has
            # it's own range of values, for every model and metric the values
            # are subtracted by the mean and divided by the standard deviation.
            # With this approach the similarity metrics are comparable.
            values = pd.Series(values)
            synonymRows[similarityColumn] = (values - values.mean()) / values.std()

            # A first, quick evaluation of the similarity scores: how well they
            # separate expert from layperson synonyms (gold type).
            types   = data.loc[data[classColumn].isin(synonymClasses), typeColumn
                ].reset_index(drop = True)
            grouped = synonymRows[similarityColumn][types.isin(synonymTypes)].groupby(
                types).agg([m, s]).reindex(synonymTypes)

            means = grouped[m].tolist()
            stds  = grouped[s].tolist()

            l.log(f"{embeddingModel} with {similarityMetric} Difference: " \
                f"[{expertSynonymType} - {laypersonSynonymType}]")
            l.log(f"Mean's Differnece:               {means[0] - means[1]}")
            l.log(f"Standard Deviation's Difference: {stds[0] - stds[1]}")
            l.log(f"{embeddingModel} with {similarityMetric} Values:     " \
                f"[{expertSynonymType},  {laypersonSynonymType}]")
            l.log(f"Means:                           {str(means)}")
            l.log(f"Standard Deviation:              {str(stds)}")

            ssmd = (means[0] - means[1]) / math.sqrt(
                math.pow(stds[0], 2) + math.pow(stds[1], 2))

            l.log("Strictly standardized mean difference: " + str(ssmd))


    # --------------------------------------------------------------------------
    # Persist data to disk: the synonyms with their similarities, which
    # syntypeeval.py joins to the classified synonyms.
    # --------------------------------------------------------------------------

    writeCSV(synonymRows, outputFileEmbedded)
    writeSettings(outputFileEmbedded, embedSettings)

    # For time tracking.
    minutes         = int((time.time() - startTime) // 60)

    # Printing Footer with minutes needed for the job.
    l.printHeader(f"Embedding of HPO Terms completed [Minutes: {minutes}]")

# ==============================================================================
# Classification of the synonyms with the LLM -> the raw classification file.
# ==============================================================================

def classifySynonyms(modelID : str) -> pd.DataFrame:
    """
    Classifies the synonyms with one model and returns them with its answers.
    """
    from model              import Model
    from model.ModelUtils   import (
        messageRoleElement,
        messageTextElement,
        modelRole,
        userRole
    )

    logging.getLogger("vllm").setLevel(logging.ERROR)

    modelName = modelNameOf(modelID)
    rawFile   = outputFileClass(modelName)

    l.printHeader(f"Classifying the Synonyms ({modelName})")

    # To track time.
    startTime = time.time()

    # Only proceed if transformed input data exists
    exitIfFileNotExist(inputFileClass)

    gold     = readCSV(inputFileClass)

    # Filter the synonyms of the concepts to classify.
    synonyms = gold[gold[classColumn].isin(synonymClasses)].copy().reset_index(
        drop = True)
    hpoIDs   = getHPOIDs(synonyms)
    synonyms = synonyms[synonyms[hpoidColumn].isin(hpoIDs)].copy().reset_index(
        drop = True)

    # --------------------------------------------------------------------------
    # Concept information for the prompts.
    # --------------------------------------------------------------------------

    def elements(className : str, key : str = hpoidColumn,
            value : str = contentColumn) -> dict:
        """
        The values of the rows of a class per key, in the order of the data.
        """
        rows = gold[gold[classColumn] == className]
        return rows.groupby(key, sort = False)[value].agg(list).to_dict()

    labels      = elements(labelClass)
    definitions = elements(definitionClass)
    comments    = elements(commentClass)
    # The children rows hold the ID of a child concept as content.
    childIDs    = elements(childrenClass)
    parentIDs   = elements(childrenClass, key = contentColumn, value = hpoidColumn)

    def relatedLabels(relatedIDs : dict, hpoID : str) -> list:
        """
        The labels of the parents or children of a concept.
        """
        return [label for i in relatedIDs.get(hpoID, []) for label in labels.get(i, [])]

    def text(values : dict, hpoID : str) -> str:
        return "".join(str(v) for v in values.get(hpoID, []))

    # --------------------------------------------------------------------------
    # Create the prompts.
    # --------------------------------------------------------------------------

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
                    text(labels, hpoID),
                    text(definitions, hpoID),
                    text(comments, hpoID),
                    relatedLabels(parentIDs, hpoID),
                    relatedLabels(childIDs, hpoID)
                ))
            else:
                messages.append(sourceClassificationPrompt(
                    text(labels, hpoID),
                    text(definitions, hpoID),
                    text(comments, hpoID),
                    relatedLabels(parentIDs, hpoID),
                    relatedLabels(childIDs, hpoID),
                    row[contentColumn]
                ))

            progress.update(task, advance = 1)

        progress.refresh()

    # With a single prompt its answer is the final answer.
    synonyms["{}{}".format(userRole, 1)]  = messages
    if chainOfThoughts:
        synonyms["{}{}".format(modelRole, 1)] = [""] * len(synonyms.index)
    else:
        synonyms[answerColumn]                = [""] * len(synonyms.index)

    # Here the other prompts for the Chain-Of-Thoughts approach are added.
    # The Few-Shot approach is incorporated directl into the prompts.
    if chainOfThoughts:
        synonyms["{}{}".format(userRole, 2)]  = [
            sourceClassificationPrompt2(synonym) for synonym in synonyms[contentColumn]]
        synonyms["{}{}".format(modelRole, 2)] = [""] * len(synonyms.index)
        synonyms["{}{}".format(userRole, 3)]  = [sourceClassificationPrompt3()] * len(synonyms.index)
        synonyms[answerColumn]                = [""] * len(synonyms.index)

    synonyms[systemColumn] = [modelName] * len(synonyms.index)

    # --------------------------------------------------------------------------
    # The answers of an earlier run are kept if it had the same prompts.
    # --------------------------------------------------------------------------

    checkpointFile = os.path.join(configurationFile["llm"]["prompt_tmp_folder"],
        checkpointFileNameClass(modelName))

    if isFile(rawFile):
        earlier = readCSV(rawFile)
        columns = [hpoidColumn, contentColumn] + [
            c for c in synonyms.columns if c.startswith(userRole)]

        if (len(earlier.index) == len(synonyms.index) and
            all(c in earlier.columns for c in columns) and
            earlier[columns].astype(str).equals(synonyms[columns].astype(str))):
            l.log("Continuing the earlier run with the same prompts.")
            synonyms = earlier
        else:
            l.error(f"{os.path.basename(rawFile)} holds a run with other " \
                "synonyms or prompts. Delete it and the checkpoint " \
                f"{os.path.basename(checkpointFile)} to start again.")
            sys.exit(1)
    else:
        l.log("Write Prompts to file.")
        writeHugeCSV(synonyms, rawFile)

    l.log(f"Few-Shot (Zero-Shot): {fewShot}")
    l.log(f"Chain-Of-Thoughts: {chainOfThoughts}")
    l.log(f"Add Definitions to Prompt: {addDefinition}")
    l.log(f"Add Comment to Prompt: {addComment}")
    l.log(f"Add Parents to Prompt: {addParents}")
    l.log(f"Add Children to Prompt: {addChildren}")

    # --------------------------------------------------------------------------
    # Set up the LLM.
    # --------------------------------------------------------------------------

    # Every model and experiment needs its own checkpoint and prompt log, so
    # these are set on top of the configuration file.
    configuration["llm"]["model_id"]        = [modelID]
    configuration["llm"]["prompt_tmp_file"] = checkpointFileNameClass(modelName)
    configuration["llm"]["prompt_log_file"] = promptLogFileName(modelName)
    # Recorded with their hashes in the run metadata, and naming the columns of
    # the checkpoint.
    configuration["llm"]["prompt_files"]    = promptFiles

    def hasText(value) -> bool:
        """
        Whether a value read from the CSV file is a non-empty string.
        """
        return isinstance(value, str) and value != ""

    def userTurns(model : Model) -> int:
        """
        Number of user prompts in the histories (all histories have the same).
        """
        histories = model.getMessageHistories()
        if len(histories) == 0:
            return 0
        return sum(1 for message in histories[0] if message[messageRoleElement] == userRole)

    def turnAnswer(history : list, turn : int) -> str:
        """
        The answer to the user prompt of the given turn (counted from 0), or ""
        if it has none (prompt longer than max_model_len).
        """
        prompts = [
            index for index, message in enumerate(history)
            if message[messageRoleElement] == userRole
        ]
        index = prompts[turn] + 1
        if index >= len(history) or history[index][messageRoleElement] != modelRole:
            return ""
        return str(history[index][messageTextElement]).strip()

    # One (prompt column, answer column) pair per turn of the conversation; the
    # answer of the last turn is the final answer. The Few-Shot approach is
    # incorporated directly into the prompts.
    turns = [("{}{}".format(userRole, 1), answerColumn)]
    if chainOfThoughts:
        turns = [
            ("{}{}".format(userRole, 1), "{}{}".format(modelRole, 1)),
            ("{}{}".format(userRole, 2), "{}{}".format(modelRole, 2)),
            ("{}{}".format(userRole, 3), answerColumn)
        ]

    # --------------------------------------------------------------------------
    # Classification of synonyms.
    # --------------------------------------------------------------------------

    l.log(f"Set up the LLM ({modelName})...")
    with Model(configuration) as model:
        l.log(f"Set up of LLM complete.")

        # An interrupted run is resumed from the model's checkpoint, which holds
        # the turns generated so far, including part of the current one.
        try:
            model.importHistoriesFromFile()
            l.log(f"Resuming from the checkpoint ({userTurns(model)} turn(s)).")
        except FileNotFoundError:
            pass
        except ValueError as e:
            # E.g. a checkpoint written before the prompts were read from
            # prompt/, whose columns are named differently.
            l.error(f"The checkpoint {os.path.basename(checkpointFile)} cannot be " \
                f"read ({e}). Delete it to start this run again; the answers " \
                f"completed in {os.path.basename(rawFile)} are kept.")
            sys.exit(1)

        for turn, (promptColumn, targetColumn) in enumerate(turns):
            l.log(f"({turn + 1}) Generating.")

            if userTurns(model) <= turn:
                model.addPrompt(userRole, synonyms[promptColumn].tolist())

                # Without a checkpoint, the answers of a turn completed earlier
                # are taken from the CSV file instead of being generated again.
                answers = synonyms[targetColumn].tolist()
                if all(hasText(answer) for answer in answers):
                    model.addPrompt(modelRole, answers)
                    l.log("Skipping Generation.")
                    continue

            # Generates the answers missing in this turn; histories that already
            # have one (after resuming) are skipped.
            model.generate()

            synonyms[targetColumn] = [
                turnAnswer(history, turn) for history in model.getMessageHistories()
            ]
            writeHugeCSV(synonyms, rawFile)

        # The final answers are JSON objects with the classification and its
        # confidence, parsed by the model package.
        l.log("Parsing the answers...")
        parsed = [
            parseClassification(model.toJSON(index))
            for index in range(len(model.getMessageHistories()))
        ]
        synonyms[synonymClass]     = [classification for classification, _ in parsed]
        synonyms[confidenceColumn] = [confidence for _, confidence in parsed]
        l.log(f"{sum(c != undefinedSynonymType for c in synonyms[synonymClass])} of " \
            f"{len(parsed)} answers contain a valid classification.")
        writeHugeCSV(synonyms, rawFile)

    # For time tracking.
    minutes         = int((time.time() - startTime) // 60)

    # Print a formatted header indicating the end of this processing stage
    l.printHeader(f"Synonyms Classified [Minutes: {minutes}]")

    return synonyms

# ==============================================================================
# Formatting of the answers -> the formatted classification file (merged by
# syntypemerge.py).
# ==============================================================================

def formatAnswers(synonyms : pd.DataFrame, modelName : str) -> None:
    l.printHeader(f"Formatting Answers of Synonym's Classification ({modelName})")

    # To track time.
    startTime = time.time()

    # classifySynonyms() parsed the answers with the model package
    # (model.toJSON()) into a classification and a confidence; the formatted
    # answer is the classification.
    classified = synonyms.copy()
    classified[answerColumn]     = classified[synonymClass].fillna(undefinedSynonymType)
    classified[confidenceColumn] = classified[confidenceColumn].fillna(-1).astype(int)
    classified                   = classified.drop(columns = synonymClass)

    # --------------------------------------------------------------------------
    # Persist transformed data to disk.
    # --------------------------------------------------------------------------

    writeCSV(classified, outputFileClassFormatted(modelName))

    l.log("Logging incorrect classified Synonyms...")

    # For logging purposes the gold standard is read and wrong classifications
    # are placed in the logging file. This is useful when it comes to prompt 
    # optimization.
    gold   = readCSV(inputFileClass)
    labels = gold[gold[classColumn] == labelClass].copy().reset_index(drop = True)
    count  = 0

    for index, row in classified.iterrows():
        # It should log, when:
        # - The answer could not be formatted e.g. is undefined.
        # - The gold type is one of the types the LLM classifies into (expert,
        #       layperson) and the answer differs from it.
        if (str(row[answerColumn]).lower() == undefinedSynonymType.lower() or
            (str(row[typeColumn]).lower() in synonymTypes and
            str(row[answerColumn]).lower() != str(row[typeColumn]).lower())):
            l.log("Label: " \
                f"{applyFormat(getElements(labels, row[hpoidColumn], labelClass))}" \
                f", Synonym: {quote(row[contentColumn])}, " \
                f"Correct: {quote(row[typeColumn])}, Classified: " \
                f"{quote(row[answerColumn])}", cmdline = False)
            count = count + 1

    l.log(f"Incorrect Classifications: {count}")
    l.log(f"Correct Classifications:   {len(classified.index) - count}")
    l.log("Logging competed.")

    # For time tracking.
    minutes         = int((time.time() - startTime) // 60)

    # Printing Footer with minutes needed for the job.
    l.printHeader(f"Formatting completed [Minutes: {minutes}]")

# ==============================================================================
# The steps of this run.
# ==============================================================================

# The HPO is transformed once, by the first of the runs started at the same
# time that finds it missing, older than hp.owl or made with other settings;
# the others wait for it. The reason is logged after the lock is taken.
if transformOutdated(log = False):
    with open(outputFileTransformedFull + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if transformOutdated():
            transformHPO()
        fcntl.flock(lock, fcntl.LOCK_UN)
else:
    l.log("The transformed HPO is up to date.")

if len(modelIDs) == 0:
    # Preparation run: the embeddings of the evaluation.
    if outdated(outputFileEmbedded, [outputFileTransformedFull], embedSettings):
        embedTerms()
    else:
        l.log("The embeddings are up to date.")
else:
    # The models classify one after the other; a failing model does not stop
    # the others. Each model is unloaded from the GPU before the next one.
    failed = []
    for index, modelID in enumerate(modelIDs):
        l.log(f"Model {index + 1}/{len(modelIDs)}: {quote(modelID)}")
        try:
            formatAnswers(classifySynonyms(modelID), modelNameOf(modelID))
        except (Exception, SystemExit) as e:
            l.error(f"Classifying with {quote(modelID)} failed: {e!r}")
            failed.append(modelID)

    if len(failed) > 0:
        l.error(f"{len(failed)} of {len(modelIDs)} model(s) failed: {applyFormat(failed)}.")
        sys.exit(1)
