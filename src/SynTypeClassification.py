"""
The preparation run and the classification of syntype (see the comment below
for the command line): transforms the HPO, embeds its terms for the
evaluation, classifies the synonyms with the LLMs of llm.model_id and formats
their answers.
"""

import os
import sys
import copy
import shlex
import json
import logging
import math

import numpy  as np
import pandas as pd

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from SynTypeUtils import *
from config     import *

from logger     import Logger, lockedFile, writeJSON

# The steps of syntype. The packages of a step (adapter, embedder, model)
# are imported by the step itself, so that a run only loads what it needs:
# the preparation run (no model given) does not load vLLM, the
# classification runs (start.sh) do not load the embedding models.
#
# Run from the repository root (see start.sh and classify.sh):
#
#   python3 ./src/SynTypeClassification.py
#       Preparation: transforms the HPO (if needed) and embeds its terms (if
#       needed) for the evaluation.
#   python3 ./src/SynTypeClassification.py <index>
#       Classification of the experiment syntype.experiments[<index>]:
#       transforms the HPO (if needed), then every model of llm.model_id
#       classifies the synonyms and its answers are formatted, one model after
#       the other.
#   python3 ./src/SynTypeClassification.py <index> <model ID>
#       The same with only this model.

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

# The logger takes the "logger" section of the configuration file; the
# command line of the run is its first line, since the log file is named
# after the start of the run only.
l = Logger(config = configFile)
l.log(f"Command: python3 {shlex.join(sys.argv)}")

# The adapter and the embedder take their sections from this copy of the
# configuration file, on which the embedding model of a run is set; the model
# package reads the llm section from the file itself.
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

# The transformation depends on the adapter settings, the test set and the
# format of the transformed HPO (the columns of the adapter package).
transformSettings = {
    "adapter"  : configurationFile["adapter"],
    "test_ids" : testIDs,
    "format"   : [idColumn, attributeColumn, valueColumn, additionalColumn]
}

# The embedding depends on every embedder setting except those that do not
# change the vectors, on the names and metrics of the distance columns, and
# on the mean and standard deviation they are normalised with.
embedderSettingsWithoutEffect = ["device", "backend", "num_threads",
    "batch_size", "cache_folder", "cache_decimals", "progress_interval"]
embedSettings = {
    "embedder"              : {
        key : value for key, value in configurationFile["embedder"].items()
        if key not in embedderSettingsWithoutEffect
    },
    "embedding_model_names" : syntypeConfig.embedding_model_names,
    "distance_metrics"    : distanceMetrics,
    "cosine_distance"     : [cosineDistanceMean, cosineDistanceStd]
}

def settingsFile(output : str) -> str:
    """
    The file holding the settings an output was made with.
    """
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
    """
    Records the settings an output was made with, next to it (atomically).
    """
    writeJSON(settings, settingsFile(output))

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
    """
    Whether the transformed HPO (and its test set) is missing, older than
    hp.owl or made with other settings; the reason is logged if `log`.
    """
    outputs = [outputFileTransformedFull] + ([outputFileTransformed] if len(testIDs) > 0 else [])
    return any(outdated(output, [inputFileTransformed], transformSettings, log) for output in outputs)

# ==============================================================================
# Prompts: the model loads the templates of the experiment (llm.prompt_files)
# and fills in the concept information, the synonym and the Few-Shot examples
# (fewShotExamples in config.py); this script gives their order and values.
# ==============================================================================

def createJSONExample(cl : str = "", co : str = "5", ra : str = "") -> str:
    """
    The answer of a Few-Shot example, in the output format of the prompts.
    """
    ret = "{\n" \
        f"\t{quote(synonymClass)}: {quote(cl)},\n" \
        f"\t{quote(confidenceColumn)}: {co},\n" \
        f"\t{quote('rationale')}: {json.dumps(ra)}\n" \
        "}"
    
    return ret

def formatInput(label : str = "", definition : str = "", comment : str = "", parents : list = [], children : list = [], synonym : str = "") -> str:
    """
    The input of a Few-Shot example, in the format of the input of the
    prompts: the information the experiment does not add is left empty.
    """
    values = promptValues(label, definition, comment, parents, children, synonym)

    return f"* Label: {values['hpo_label']}\n" \
        f"* Synonyms: {values['hpo_synonyms']}\n" \
        f"* Definition: {values['hpo_definition']}\n" \
        f"* Comment: {values['hpo_comment']}\n" \
        f"* Parents: {values['hpo_parents']}\n" \
        f"* Children: {values['hpo_children']}"

def getExamples() -> str:
    """
    The Few-Shot examples (fewShotExamples in config.py), each with its input
    and its answer in the format of the prompts.
    """
    ret = "Examples"

    for index, example in enumerate(fewShotExamples):
        ret += f"""

Example {index + 1}

Input:
{formatInput(example["label"], example["definition"], example["comment"], example["parents"], example["children"], example["synonym"])}

Output:
{createJSONExample(example["type"], example["confidence"], example["rationale"])}"""

    return ret

def examples(fewShot : bool = fewShot) -> str:
    """
    The value of {examples}: the Few-Shot examples, or nothing.
    """
    return "\n\n" + getExamples() if fewShot else ""

def promptValues(
        label : str = "",
        definition : str = "",
        comment : str = "",
        parents : list = [],
        children : list = [],
        synonym : str = ""
) -> dict:
    """
    The values of the placeholders of the prompts (see prompts/README.md),
    without {examples}: the information the experiment does not add is left
    empty, so that every prompt of an experiment has the same lines.
    """
    return {
        "hpo_label"      : quote(label),
        "hpo_synonyms"   : quote(synonym),
        "hpo_definition" : quote(definition) if addDefinition else "",
        "hpo_comment"    : quote(comment) if addComment else "",
        "hpo_parents"    : applyFormat(parents) if addStructure else "",
        "hpo_children"   : applyFormat(children) if addStructure else ""
    }

# The prompts of the experiment, one per turn, by their index in
# llm.prompt_files, from where the model loads them: 0 is the single prompt,
# 1 to 3 are the chain-of-thoughts prompts.
singlePromptIndex            = 0
chainOfThoughtsPromptIndexes = [1, 2, 3]

promptIndexes = chainOfThoughtsPromptIndexes if chainOfThoughts else [singlePromptIndex]

# ==============================================================================
# Transformation: hp.owl (adapter) -> transform.csv, transform.reduced.csv.
# ==============================================================================

def transformHPO() -> None:
    """
    Transforms hp.owl with the adapter into transform.csv (all concepts) and
    transform.reduced.csv (the test concepts with their parents and
    children), in the adapter's format: the conflicting synonyms and the
    synonyms equal to their label are removed. Records the settings next to
    both files.
    """
    from adapter    import HPOAdapter

    # A synonym of a concept with all classes or all types of one of these sets
    # has no unambiguous gold label and is removed.
    conflictingSynonymClasses = {exactSynonymClass, relatedSynonymClass}
    conflictingSynonymTypes   = {laypersonSynonymType, abbreviationSynonymType}

    l.printHeader("Transforming the Raw HPO Data")

    # --------------------------------------------------------------------------
    # Load the HPO with the adapter. It stops with an error if hp.owl is
    # missing, extracts labels, definitions, comments, children, references
    # and synonyms of the active (not obsolete) concepts, removes rows without
    # an ID or content and merges duplicates: a synonym listed several times
    # for a concept is one row whose classes and types are joined, e.g.
    # "exact, related".
    # --------------------------------------------------------------------------

    adapter = HPOAdapter(configuration)
    adapter.load()

    # --------------------------------------------------------------------------
    # Remove the synonyms listed both as exact and related synonym, or both as
    # layperson synonym and abbreviation (an expert term), which have no
    # unambiguous gold label. The others count with their first class and type
    # (synonymClassOf(), synonymTypeOf()).
    # --------------------------------------------------------------------------

    data       = adapter.data
    additional = data.loc[data[attributeColumn] == synonymAttribute, additionalColumn]
    classes    = additional.map(lambda a: set(additionalValues(a, semanticClass)))
    types      = additional.map(lambda a: set(additionalValues(a, sourceType)))

    conflictingClasses = classes.map(lambda c: conflictingSynonymClasses <= c)
    conflictingTypes   = types.map(lambda t: conflictingSynonymTypes <= t)
    l.log(f"Removing {conflictingClasses.sum()} synonyms listed as both " \
        f"{applyFormat(sorted(conflictingSynonymClasses))} and " \
        f"{(conflictingTypes & ~conflictingClasses).sum()} further synonyms " \
        f"listed as both {applyFormat(sorted(conflictingSynonymTypes))}...")
    data = data.drop(additional.index[(conflictingClasses | conflictingTypes).to_numpy()])

    # --------------------------------------------------------------------------
    # Remove the synonyms that match the label of their concept (ignoring
    # case).
    # --------------------------------------------------------------------------

    rowCount = len(data.index)
    l.log("Removing synonyms having a match with labels...")
    terms   = data[idColumn].astype(str) + data[valueColumn].astype(str).str.lower()
    matches = synonymRows(data) & terms.isin(terms[data[attributeColumn] == labelClass])
    data    = data[~matches].reset_index(drop = True)
    l.log(f"Removed {rowCount - len(data.index)} synonyms due to having a " \
        "match with their label.")

    l.printRowCount(data)
    printDataSummary(data)

    l.log("Write full transformated Data...")
    writeHugeCSV(data, outputFileTransformedFull)
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
            (data[valueColumn].isin(testIDs)) &
            (data[attributeColumn] == childrenClass),
            idColumn
        ].tolist()

        childIDs = data.loc[
            (data[idColumn].isin(testIDs)) &
            (data[attributeColumn] == childrenClass),
            valueColumn
        ].tolist()

        hpoIDs = list(set(testIDs + childIDs + parentIDs))

        # ----------------------------------------------------------------------
        # Reduce content to only the selected HPO concepts.
        # ----------------------------------------------------------------------

        result = data[data[idColumn].isin(hpoIDs)].copy().reset_index(
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
        writeHugeCSV(result, outputFileTransformed)
        writeSettings(outputFileTransformed, transformSettings)
        l.log("Reduced transformed Data written.")

# ==============================================================================
# Embedding: the cosine distance of every synonym to its label -> embed.csv,
# for the evaluation (SynTypeEval.py).
# ==============================================================================

def embedTerms() -> None:
    """
    Embeds the labels and synonyms of all concepts with every model of
    embedder.model_id and writes the normalised cosine distance of every
    synonym to its label to embed.csv, for the evaluation. Logs how well the
    distance separates expert from layperson synonyms.
    """
    from embedder import Embedder

    # Printing header.
    l.printHeader("Embedding HPO Terms")

    # --------------------------------------------------------------------------
    # Load the Human Phenotype Ontology (HPO) from the transformed data file.
    # All concepts are embedded, also for test runs: the distances are
    # standardised with one mean and standard deviation (configured, or of all
    # synonyms), so that test and full runs share one scale.
    # --------------------------------------------------------------------------

    # Ensure the input file exists before proceeding.
    exitIfFileNotExist(outputFileTransformedFull)

    data = readTransformed(outputFileTransformedFull)

    l.log("Filtering Labels and Synonyms of HPO Concepts...")
    # One label per concept (the first, if a concept has several).
    labelRows   = data[data[attributeColumn] == labelClass].drop_duplicates(
        subset = idColumn)
    labels      = labelRows[valueColumn].astype(str).tolist()

    synonymRows = synonymTable(data)
    types       = synonymRows[typeColumn]
    synonymRows = synonymRows[[hpoidColumn, classColumn, contentColumn]].copy()
    synonyms    = synonymRows[contentColumn].astype(str).tolist()

    # Position of the label of each synonym's concept in labels (-1 if the
    # concept has no label).
    labelIndex  = synonymRows[hpoidColumn].map(
        pd.Series(range(len(labels)), index = labelRows[idColumn])
    ).fillna(-1).astype(int).to_numpy()
    hasLabel    = labelIndex >= 0
    l.log(f"Found {len(labels)} labels and {len(synonyms)} synonyms " \
        f"({(~hasLabel).sum()} of a concept without label).")

    def distance(a : np.ndarray, b : np.ndarray) -> np.ndarray:
        """
        Cosine distance of the rows of a and b: 1 minus their cosine
        similarity, from 0 (same direction) to 2 (opposite directions).
        """
        return 1 - np.sum(a * b, axis = 1) / (
            np.linalg.norm(a, axis = 1) * np.linalg.norm(b, axis = 1))

    # --------------------------------------------------------------------------
    # Embed and calculate the distance of labels to their synonyms.
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

        for distanceMetric in distanceMetrics:
            distanceColumn = distanceColumnPrefix.format(embeddingModel,
                distanceMetric)

            # Standard value for no distance (no label) is np.nan.
            values           = np.full(len(synonyms), np.nan)
            values[hasLabel] = distance(a, b)

            # Normalization: the distances are subtracted by the mean and
            # divided by the standard deviation, those of
            # syntype.cosine_distance_mean and _std if set, so that every run
            # uses the same scale, otherwise those of the data. Normalised
            # distances can be below -1 and above 1.
            values = pd.Series(values)
            mean   = values.mean() if cosineDistanceMean is None else cosineDistanceMean
            std    = values.std() if cosineDistanceStd is None else cosineDistanceStd
            l.log(f"{embeddingModel} with {distanceMetric}: mean {values.mean()} and " \
                f"standard deviation {values.std()} of the data, normalised with " \
                f"mean {mean} and standard deviation {std}.")
            synonymRows[distanceColumn] = (values - mean) / std

            # A first, quick evaluation of the distances: how well they
            # separate expert from layperson synonyms (gold type).
            grouped = synonymRows[distanceColumn][types.isin(synonymTypes)].groupby(
                types).agg([m, s]).reindex(synonymTypes)

            means = grouped[m].tolist()
            stds  = grouped[s].tolist()

            l.log(f"{embeddingModel} with {distanceMetric} Difference: " \
                f"[{expertSynonymType} - {laypersonSynonymType}]")
            l.log(f"Mean's Difference:               {means[0] - means[1]}")
            l.log(f"Standard Deviation's Difference: {stds[0] - stds[1]}")
            l.log(f"{embeddingModel} with {distanceMetric} Values:     " \
                f"[{expertSynonymType},  {laypersonSynonymType}]")
            l.log(f"Means:                           {str(means)}")
            l.log(f"Standard Deviation:              {str(stds)}")

            ssmd = (means[0] - means[1]) / math.sqrt(
                math.pow(stds[0], 2) + math.pow(stds[1], 2))

            l.log("Strictly standardized mean difference: " + str(ssmd))


    # --------------------------------------------------------------------------
    # Persist data to disk: the synonyms with their distances, which
    # SynTypeEval.py joins to the classified synonyms.
    # --------------------------------------------------------------------------

    writeHugeCSV(synonymRows, outputFileEmbedded)
    writeSettings(outputFileEmbedded, embedSettings)

# ==============================================================================
# Classification of the synonyms with the LLM -> the raw classification file.
# ==============================================================================

def classifySynonyms(modelID : str) -> pd.DataFrame:
    """
    Classifies the synonyms with one model and returns them with its answers.
    """
    from model              import Model
    from model.ModelUtils   import modelRole, userRole

    logging.getLogger("vllm").setLevel(logging.ERROR)

    # The model is loaded by its index in llm.model_id.
    if modelID not in configurationFile["llm"]["model_id"]:
        raise ValueError(f"{quote(modelID)} is not in llm.model_id.")
    modelIndex = configurationFile["llm"]["model_id"].index(modelID)

    modelName = modelNameOf(modelID)
    rawFile   = outputFileClass(modelName)

    l.printHeader(f"Classifying the Synonyms ({modelName})")

    # Only proceed if transformed input data exists
    exitIfFileNotExist(inputFileClass)

    gold     = readTransformed(inputFileClass)

    # The synonyms of the concepts to classify, with their gold type.
    synonyms = synonymTable(gold)
    hpoIDs   = getHPOIDs(synonyms)
    synonyms = synonyms[synonyms[hpoidColumn].isin(hpoIDs)].copy().reset_index(
        drop = True)

    # --------------------------------------------------------------------------
    # Concept information for the prompts.
    # --------------------------------------------------------------------------

    def elements(attribute : str, key : str = idColumn,
            value : str = valueColumn) -> dict:
        """
        The values of the rows of an attribute per key, in the order of the
        data.
        """
        rows = gold[gold[attributeColumn] == attribute]
        return rows.groupby(key, sort = False)[value].agg(list).to_dict()

    labels      = elements(labelClass)
    definitions = elements(definitionClass)
    comments    = elements(commentClass)
    # The child rows hold the ID of a child concept as value.
    childIDs    = elements(childrenClass)
    parentIDs   = elements(childrenClass, key = valueColumn, value = idColumn)

    def relatedLabels(relatedIDs : dict, hpoID : str) -> list:
        """
        The labels of the parents or children of a concept.
        """
        return [label for i in relatedIDs.get(hpoID, []) for label in labels.get(i, [])]

    def text(values : dict, hpoID : str) -> str:
        """
        The values of a concept joined into one text ("" if it has none).
        """
        return "".join(str(v) for v in values.get(hpoID, []))

    # --------------------------------------------------------------------------
    # The values of the placeholders of the prompts, one dictionary per synonym
    # (the model loads the prompts and fills them in).
    # --------------------------------------------------------------------------

    examplesValue = examples()
    values = [
        promptValues(
            text(labels, hpoID),
            text(definitions, hpoID),
            text(comments, hpoID),
            relatedLabels(parentIDs, hpoID),
            relatedLabels(childIDs, hpoID),
            synonym
        ) | {"examples" : examplesValue}
        for hpoID, synonym in zip(synonyms[hpoidColumn], synonyms[contentColumn])
    ]

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
    # The raw classification file: the synonyms with the prompts and answers of
    # every turn, written after every turn.
    # --------------------------------------------------------------------------

    for promptColumn, targetColumn in turns:
        synonyms[promptColumn] = [""] * len(synonyms.index)
        synonyms[targetColumn] = [""] * len(synonyms.index)
    synonyms[systemColumn] = [modelName] * len(synonyms.index)

    l.log(f"Few-Shot (Zero-Shot): {fewShot}")
    l.log(f"Chain-Of-Thoughts: {chainOfThoughts}")
    l.log(f"Add Definitions to Prompt: {addDefinition}")
    l.log(f"Add Comment to Prompt: {addComment}")
    l.log(f"Add Parents and Children to Prompt: {addStructure}")

    # --------------------------------------------------------------------------
    # Classification of synonyms.
    # --------------------------------------------------------------------------

    # The model is loaded by its index in llm.model_id, with the llm section as
    # it is. Its answer store (the checkpoint) and prompt log are named after
    # the model and the experiment (the postfix), since start.sh starts many
    # runs at the same time.
    l.log(f"Set up the LLM ({modelName})...")
    with Model(configFile, modelIndex) as model:
        model.setPostfix(f"{testStr}_{configStr}_{chainOfThoughtsStr}_{fewShotStr}")
        l.log(f"Set up of LLM complete.")

        # The model uses the answers in its answer store and only generates the
        # others, so an interrupted run resumes where it stopped; without the
        # store the classification starts anew. A changed prompt has no stored
        # answers, and a store written with other model settings is refused.
        for turn, (promptColumn, targetColumn) in enumerate(turns):
            l.log(f"({turn + 1}) Generating.")

            # The model loads the prompt of this turn (by its index in
            # llm.prompt_files) and fills in its placeholders.
            model.addPromptsFromFile(promptIndexes[turn], values)
            synonyms[promptColumn] = [model.getPrompt(index, turn) for index in range(len(values))]

            # Generates the answers that are not in the answer store.
            model.generate()

            synonyms[targetColumn] = [model.getAnswer(index, turn) for index in range(len(values))]
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

        # Every prompt and answer of this model and experiment.
        l.log("Writing the prompts and answers to the prompt log...")
        model.logPrompts()

    return synonyms

# ==============================================================================
# Formatting of the answers -> the formatted classification file (merged by
# SynTypeMerge.py).
# ==============================================================================

def formatAnswers(synonyms : pd.DataFrame, modelName : str) -> None:
    """
    Writes the formatted classification of a model (its parsed answers, see
    classifySynonyms()), which SynTypeMerge.py merges, and logs every wrong or
    undefined classification.
    """
    l.printHeader(f"Formatting Answers of Synonym's Classification ({modelName})")

    # classifySynonyms() parsed the answers with the model package
    # (model.toJSON()) into a classification and a confidence; the formatted
    # answer is the classification.
    classified = synonyms.copy()
    classified[answerColumn]     = classified[synonymClass].fillna(undefinedSynonymType)
    classified[confidenceColumn] = classified[confidenceColumn].fillna(-1).astype(int)
    classified                   = classified.drop(columns = synonymClass)

    # --------------------------------------------------------------------------
    # Persist the formatted classification to disk.
    # --------------------------------------------------------------------------

    writeHugeCSV(classified, outputFileClassFormatted(modelName))

    l.log("Logging incorrect classified Synonyms...")

    # For logging purposes the gold standard is read and wrong classifications
    # are placed in the logging file. This is useful when it comes to prompt 
    # optimization.
    gold   = readTransformed(inputFileClass)
    labels = gold[gold[attributeColumn] == labelClass].copy().reset_index(drop = True)
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

# ==============================================================================
# The steps of this run.
# ==============================================================================

# The HPO is transformed once, by the first of the runs started at the same
# time that finds it missing, older than hp.owl or made with other settings;
# the others wait for it. The reason is logged after the lock is taken.
if transformOutdated(log = False):
    with lockedFile(outputFileTransformedFull):
        if transformOutdated():
            with l.step("Transforming the HPO"):
                transformHPO()
else:
    l.log("The transformed HPO is up to date.")

if experimentIndex is None:
    # Preparation run: the embeddings of the evaluation.
    if outdated(outputFileEmbedded, [outputFileTransformedFull], embedSettings):
        with l.step("Embedding the HPO terms"):
            embedTerms()
    else:
        l.log("The embeddings are up to date.")
else:
    # The models classify one after the other; a failing model does not stop
    # the others. Each model is unloaded from the GPU before the next one. The
    # logger logs the start, end and duration of every step.
    failed = []
    for index, modelID in enumerate(modelIDs):
        l.log(f"Model {index + 1}/{len(modelIDs)}: {quote(modelID)}")
        modelName = modelNameOf(modelID)
        try:
            with l.step(f"Classifying the synonyms ({modelName})"):
                synonyms = classifySynonyms(modelID)
            with l.step(f"Formatting the answers ({modelName})"):
                formatAnswers(synonyms, modelName)
        except (Exception, SystemExit) as e:
            l.error(f"Classifying with {quote(modelID)} failed: {e!r}")
            failed.append(modelID)

    if len(failed) > 0:
        l.error(f"{len(failed)} of {len(modelIDs)} model(s) failed: {applyFormat(failed)}.")
        sys.exit(1)
