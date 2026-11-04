"""
The preparation run and the classification of syntype (see the comment below
for the command line): transforms the HPO, draws the development set,
embeds the development set and the HPO for the evaluation, classifies the synonyms with the LLMs of llm.model_id and formats
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
#       Preparation: transforms the HPO, draws the development set and
#       embeds the development set and the HPO for the evaluation, each only
#       if needed.
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

# ------------------------------------------------------------------------------
# The settings an output was made with are stored next to it
# (<output>.settings.json), so that it is made again when they change, not
# only when one of its input files is newer.
# ------------------------------------------------------------------------------

# The transformation depends on the adapter settings and the format of the
# transformed HPO (the columns of the adapter package).
transformSettings = {
    "adapter"  : configurationFile["adapter"],
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

def transformOutdated(log : bool = True) -> bool:
    """
    Whether the transformed HPO is missing, older than hp.owl or made with
    other settings; the reason is logged if `log`.
    """
    return outdated(outputFileTransformedFull, [inputFileTransformed], transformSettings, log)

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
        f"\t{quote(rationaleColumn)}: {json.dumps(ra)}\n" \
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
# Transformation: hp.owl (adapter) -> hpo.csv.
# ==============================================================================

def transformHPO() -> None:
    """
    Transforms hp.owl with the adapter into hpo.csv, in the adapter's
    format: the conflicting synonyms and the synonyms equal to their label
    are removed. Records the settings next to the file.
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

    # The other synonyms listed several times count with their first class
    # and type, which becomes their gold label.
    kept     = ~(conflictingClasses | conflictingTypes)
    severals = [
        (key, values[kept & (values.map(len) > 1)])
        for key, values in [(semanticClass, classes), (sourceType, types)]
    ]
    for key, values in severals:
        joined = additional[values.index].map(lambda a: str(a.get(key, "")))
        combinations = joined.value_counts()
        l.log(f"{len(values)} synonyms have several values of {quote(key)}; they " \
            "count with the first one" + ("." if len(values) == 0 else ": " + ", ".join(
                f"{quote(value)} {count}" for value, count in combinations.head(5).items()) +
            (", ..." if len(combinations) > 5 else "") + "."))

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

    l.log("Writing the transformed HPO...")
    writeHugeCSV(data, outputFileTransformedFull)
    writeSettings(outputFileTransformedFull, transformSettings)
    l.log("Full transformed Data written.")

# ==============================================================================
# Embedding: the cosine distance of every synonym to its label -> embed.csv
# (the evaluation set) and embed.development.csv (the development set), for the
# evaluation (SynTypeEval.py).
# ==============================================================================

def embedTerms(inputFile : str, outputFile : str, developmentSet : bool) -> None:
    """
    Embeds the synonyms of inputFile and the labels of their concepts with
    every model of embedder.model_id and writes the normalised cosine distance
    of every synonym to its label to outputFile, for the evaluation. Logs the
    mean and standard deviation of the distances, which, for the development
    set, are to be set as syntype.cosine_distance_mean and _std, and how well
    the distance separates expert from layperson synonyms.
    """
    from embedder import Embedder

    # Printing header.
    l.printHeader("Embedding the Development Set" if developmentSet else "Embedding the Evaluation Set")

    # Ensure the input file exists before proceeding.
    exitIfFileNotExist(inputFile)

    data = readTransformed(inputFile)

    l.log("Filtering Labels and Synonyms of HPO Concepts...")
    synonymRows = synonymTable(data)
    types       = synonymRows[typeColumn]
    synonymRows = synonymRows[[hpoidColumn, classColumn, contentColumn]].copy()
    synonyms    = synonymRows[contentColumn].astype(str).tolist()

    # One label per concept with synonyms (the first, if a concept has
    # several); the labels of the parents and children in the development set
    # are not needed.
    labelRows   = data[(data[attributeColumn] == labelClass)
        & data[idColumn].isin(synonymRows[hpoidColumn])].drop_duplicates(subset = idColumn)
    labels      = labelRows[valueColumn].astype(str).tolist()

    # Position of the label of each synonym's concept in labels (-1 if the
    # concept has no label).
    labelIndex  = synonymRows[hpoidColumn].map(
        pd.Series(range(len(labels)), index = labelRows[idColumn])
    ).fillna(-1).astype(int).to_numpy()
    hasLabel    = labelIndex >= 0
    l.log(f"Found {len(labels)} labels and {len(synonyms)} synonyms " \
        f"({(~hasLabel).sum()} of a concept without label).")

    def describe(values : pd.Series) -> str:
        """
        Mean, standard deviation, minimum and maximum of the values that are
        not NaN, for the log.
        """
        values = values.dropna()
        return f"mean {values.mean():.4f}, std {values.std():.4f}, " \
            f"min {values.min():.4f}, max {values.max():.4f}"

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
            normalised = (values - mean) / std
            synonymRows[distanceColumn] = normalised

            # The distances of all synonyms with a label, before and after the
            # normalization, and per gold type.
            l.log(f"{embeddingModel}, {distanceMetric} distance to the label: " \
                f"{values.notna().sum()} synonyms ({values.isna().sum()} without label).")
            l.log(f"All: {describe(values)}.")
            for synonymType in synonymTypes:
                ofType = types == synonymType
                l.log(f"Gold type {quote(synonymType)} ({ofType.sum()}): {describe(values[ofType])}.")
            l.log(f"Normalised with mean {mean} and std {std}: {describe(normalised)}.")
            for synonymType in synonymTypes:
                ofType = types == synonymType
                l.log(f"Normalised, gold type {quote(synonymType)}: {describe(normalised[ofType])}.")

            # The mean and standard deviation of the development set, with
            # which the distances of all synonyms are to be normalised.
            if developmentSet:
                l.log(f"Set syntype.cosine_distance_mean: {values.mean():.4f} and " \
                    f"cosine_distance_std: {values.std():.4f} (now {cosineDistanceMean} " \
                    f"and {cosineDistanceStd}).")

            # A first, quick evaluation of the distances: how well they
            # separate expert from layperson synonyms, as the strictly
            # standardized mean difference of the normalised distances.
            grouped = normalised[types.isin(synonymTypes)].groupby(
                types).agg([m, s]).reindex(synonymTypes)
            means = grouped[m].tolist()
            stds  = grouped[s].tolist()
            ssmd = (means[0] - means[1]) / math.sqrt(
                math.pow(stds[0], 2) + math.pow(stds[1], 2))
            l.log(f"Strictly standardized mean difference [{expertSynonymType} - " \
                f"{laypersonSynonymType}]: {ssmd:.4f} (difference of the means " \
                f"{means[0] - means[1]:.4f}, of the stds {stds[0] - stds[1]:.4f}).")


    # --------------------------------------------------------------------------
    # Persist data to disk: the synonyms with their distances, which
    # SynTypeEval.py joins to the classified synonyms.
    # --------------------------------------------------------------------------

    writeHugeCSV(synonymRows, outputFile)
    writeSettings(outputFile, embedSettings)

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

    # The synonyms to classify, with their gold type: all of the HPO, or those
    # of the development set ("test").
    synonyms = synonymTable(gold)

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

    # The setup of this run: the experiment, its prompts and its synonyms.
    settings = [name for name, value in experiment.model_dump().items() if value]
    l.log(f"Experiment {experimentIndex}: {applyFormat(settings)} " \
        f"(files {quote(filesPrefixApproachClass)}).")
    l.log("Prompt files: " + applyFormat(
        [configurationFile["llm"]["prompt_files"][index] for index in promptIndexes]))
    goldTypes = synonyms[typeColumn].value_counts()
    l.log(f"Classifying {len(synonyms.index)} synonyms of " \
        f"{synonyms[hpoidColumn].nunique()} concepts; gold types: " + ", ".join(
            f"{goldType} {count}" for goldType, count in goldTypes.items()) + ".")

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
            # The prompt of the first synonym, to see what the model is given.
            l.log(f"Prompt of turn {turn + 1} for the first synonym " \
                f"({quote(synonyms[contentColumn].iloc[0])}):\n{model.getPrompt(0, turn)}",
                cmdline = False)
            synonyms[promptColumn] = [model.getPrompt(index, turn) for index in range(len(values))]

            # Generates the answers that are not in the answer store.
            model.generate()

            synonyms[targetColumn] = [model.getAnswer(index, turn) for index in range(len(values))]
            writeHugeCSV(synonyms, rawFile)

        # The final answers are JSON objects with the classification, its
        # confidence and a rationale, parsed by the model package.
        l.log("Parsing the answers...")
        parsed = [
            parseClassification(model.toJSON(index))
            for index in range(len(model.getMessageHistories()))
        ]
        synonyms[synonymClass]     = [classification for classification, _, _ in parsed]
        synonyms[confidenceColumn] = [confidence for _, confidence, _ in parsed]
        synonyms[rationaleColumn]  = [rationale for _, _, rationale in parsed]
        l.log(f"{sum(c != undefinedSynonymType for c in synonyms[synonymClass])} of " \
            f"{len(parsed)} answers contain a valid classification, " \
            f"{sum(r != '' for r in synonyms[rationaleColumn])} a rationale.")
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

    l.log("Wrong or undefined classifications, as synonym (label): gold -> " \
        "classified (confidence): rationale.", cmdline = False)
    for index, row in classified.iterrows():
        # It should log, when:
        # - The answer could not be formatted e.g. is undefined.
        # - The gold type is one of the types the LLM classifies into (expert,
        #       layperson) and the answer differs from it.
        if (str(row[answerColumn]).lower() == undefinedSynonymType.lower() or
            (str(row[typeColumn]).lower() in synonymTypes and
            str(row[answerColumn]).lower() != str(row[typeColumn]).lower())):
            # Shortened; the full texts are in the raw and formatted files.
            label = ", ".join(getElements(labels, row[hpoidColumn], labelClass))
            l.log(f"{quote(shorten(row[contentColumn], 40))} " \
                f"(label {quote(shorten(label, 30))}): {row[typeColumn]} -> " \
                f"{row[answerColumn]} ({row[confidenceColumn]}): " \
                f"{quote(shorten(row[rationaleColumn], 60))}", cmdline = False)
            count = count + 1

    l.log(f"Incorrect Classifications: {count}")
    l.log(f"Correct Classifications:   {len(classified.index) - count}")
    l.log("Logging completed.")

    logClassificationSummary(classified, modelName)


def logClassificationSummary(classified : pd.DataFrame, modelName : str) -> None:
    """
    Logs a summary of the formatted classification of a model, as
    SynTypeEval.py scores it: of the synonyms with an expert or layperson gold
    type, without those of the Few-Shot examples, a confusion table of gold
    against classified types, the mean confidence per classified type, and
    precision, recall and F1 per type (undefined answers count as wrong).
    """
    examples = [(example[hpoidColumn], example["synonym"]) for example in fewShotExamples]
    scored = classified[
        classified[typeColumn].isin(synonymTypes) &
        ~classified[[hpoidColumn, contentColumn]].apply(tuple, axis = 1).isin(examples)
    ]
    answers = synonymTypes + [undefinedSynonymType]

    l.log(f"Summary of {modelName}: {len(scored.index)} synonyms (" + ", ".join(synonymTypes) +
        f"; without the few-shot examples), " \
        f"{(scored[answerColumn] == undefinedSynonymType).sum()} answers undefined.")
    l.log(f"{'Gold / Classified':<20}" + "".join(f"{answer:>12}" for answer in answers))
    for goldType in synonymTypes:
        ofType = scored[scored[typeColumn] == goldType]
        l.log(f"{goldType:<20}" + "".join(
            f"{(ofType[answerColumn] == answer).sum():>12}" for answer in answers))

    for synonymType in synonymTypes:
        predicted = scored[answerColumn] == synonymType
        actual    = scored[typeColumn] == synonymType
        tp = (predicted & actual).sum()
        precision = tp / predicted.sum() if predicted.sum() > 0 else float("nan")
        recall    = tp / actual.sum() if actual.sum() > 0 else float("nan")
        f1        = 2 * tp / (predicted.sum() + actual.sum()) if predicted.sum() + actual.sum() > 0 else float("nan")
        confidence = scored.loc[predicted, confidenceColumn]
        l.log(f"{quote(synonymType)}: precision {precision:.3f}, recall {recall:.3f}, " \
            f"F1 {f1:.3f}; mean confidence of the {predicted.sum()} classified as " \
            f"{synonymType}: {confidence.mean():.2f}")

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

# The development and the evaluation set are drawn from the transformed HPO,
# once by the first of the runs started at the same time that finds them
# missing or outdated.
if developmentSetOutdated(log = False):
    with lockedFile(outputFileTransformedDevelopment):
        if developmentSetOutdated():
            with l.step("Drawing the development and evaluation set"):
                writeDevelopmentSet()
else:
    l.log("The development and evaluation set are up to date.")

if experimentIndex is None:
    # Preparation run: the distances of the evaluation, of the development set
    # first, whose mean and standard deviation are logged, then of the
    # evaluation set.
    for name, inputFile, outputFile, developmentSet in [
        ("the development set", outputFileTransformedDevelopment, outputFileEmbeddedDevelopment, True),
        ("the evaluation set", outputFileTransformedEvaluation, outputFileEmbeddedEvaluation, False)
    ]:
        if outdated(outputFile, [inputFile], embedSettings):
            with l.step(f"Embedding {name}"):
                embedTerms(inputFile, outputFile, developmentSet)
        else:
            l.log(f"The distances of {name} are up to date.")
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
