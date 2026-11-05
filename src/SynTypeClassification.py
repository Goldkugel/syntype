"""
The preparation run and the classification of syntype: transforms the HPO,
draws the development set, embeds the development and the evaluation set,
classifies the synonyms with the LLMs of llm.model_id and formats their answers.
Each package (adapter, embedder, model) is imported by the step that needs it,
so the preparation run does not load vLLM and the classification runs do not
load the embedding models.

    python3 ./src/SynTypeClassification.py                  preparation
    python3 ./src/SynTypeClassification.py <index>          experiment <index>, all models
    python3 ./src/SynTypeClassification.py <index> <model>  only this model
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

sys.dont_write_bytecode = True

from SynTypeUtils import *
from config     import *

from logger     import Logger, lockedFile, writeJSON

l = Logger(config = configFile)

# Adapter and embedder read this copy (the embedding model is set on it); the
# model package reads the llm section from the file itself.
configuration = copy.deepcopy(configurationFile)

# The settings an output depends on (see outdated() in SynTypeUtils.py).
transformSettings = {
    "adapter"  : configurationFile["adapter"],
    "format"   : [idColumn, attributeColumn, valueColumn, additionalColumn]
}

# Embedder settings that do not change the vectors are left out.
embedderSettingsWithoutEffect = ["device", "backend", "num_threads",
    "batch_size", "cache_folder", "cache_decimals", "progress_interval"]
embedSettings = {
    "embedder"              : {
        key : value for key, value in configurationFile["embedder"].items()
        if key not in embedderSettingsWithoutEffect
    },
    "embedding_model_names" : syntypeConfig.embedding_model_names,
    "distance_metrics"    : distanceMetrics,
    "cosine_distance"     : [cosineDistanceMean, cosineDistanceStd],
    "format"              : [idColumn, attributeColumn, valueColumn, additionalColumn]
}

def transformOutdated(log : bool = True) -> bool:
    """
    Whether hpo.csv is missing, older than hp.owl or made with other
    settings; the reason is logged if `log`.
    """
    return outdated(outputFileTransformedFull, [inputFileTransformed], transformSettings, log)

# ==============================================================================
# Prompts: the model loads the templates (llm.prompt_files) and fills in the
# values given here.
# ==============================================================================

def createJSONExample(cl : str = "", co : str = "5", ra : str = "") -> str:
    """
    The answer of a Few-Shot example, in the output format of the prompts.
    """
    ret = "{\n" \
        f"\t{quote(classificationKey)}: {quote(cl)},\n" \
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
        f"* Synonym: {values['hpo_synonyms']}\n" \
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
    The values of the placeholders of the prompts (see prompts/README.md)
    except {examples}. Information the experiment does not add is left empty,
    so every prompt keeps the same lines.
    """
    return {
        "hpo_label"      : quote(label),
        "hpo_synonyms"   : quote(synonym),
        "hpo_definition" : quote(definition) if addDefinition else "",
        "hpo_comment"    : quote(comment) if addComment else "",
        "hpo_parents"    : applyFormat(parents) if addStructure else "",
        "hpo_children"   : applyFormat(children) if addStructure else ""
    }

# Indexes in llm.prompt_files, one per turn.
singlePromptIndex            = 0
chainOfThoughtsPromptIndexes = [1, 2, 3]

promptIndexes = chainOfThoughtsPromptIndexes if chainOfThoughts else [singlePromptIndex]

# ==============================================================================
# Transformation: hp.owl (adapter) -> hpo.csv.
# ==============================================================================

def transformHPO() -> None:
    """
    Transforms hp.owl with the adapter into hpo.csv (adapter format), without
    conflicting synonyms and synonyms equal to their label, and records the
    settings next to it.
    """
    from adapter    import HPOAdapter

    # A synonym listed with both values of one of these sets (e.g. layperson
    # and abbreviation, an expert term) has no unambiguous gold label.
    conflictingSynonymClasses = {exactSynonymClass, relatedSynonymClass}
    conflictingSynonymTypes   = {laypersonSynonymType, abbreviationSynonymType}

    l.printHeader("Transforming the Raw HPO Data")

    # The adapter keeps the active concepts and merges a synonym listed several
    # times into one row whose classes and types are joined ("exact, related").
    adapter = HPOAdapter(configuration)
    adapter.load()

    data       = adapter.data
    additional = data.loc[data[attributeColumn] == synonymAttribute, additionalColumn]
    classes    = additional.map(lambda a: set(additionalValues(a, semanticClass)))
    types      = additional.map(lambda a: set(additionalValues(a, sourceType)))

    conflictingClasses = classes.map(lambda c: conflictingSynonymClasses <= c)
    conflictingTypes   = types.map(lambda t: conflictingSynonymTypes <= t)
    l.log("Removing the synonyms with conflicting gold labels...")
    l.log(f"{conflictingClasses.sum()} synonyms listed as both " \
        f"{applyFormat(sorted(conflictingSynonymClasses))}.")
    l.log(f"{(conflictingTypes & ~conflictingClasses).sum()} further synonyms " \
        f"listed as both {applyFormat(sorted(conflictingSynonymTypes))}.")
    data = data.drop(additional.index[(conflictingClasses | conflictingTypes).to_numpy()])
    l.log("Removing the synonyms with conflicting gold labels completed.")

    # The other synonyms listed several times count with their first class and
    # type (synonymClassOf(), synonymTypeOf()).
    kept     = ~(conflictingClasses | conflictingTypes)
    severals = [
        (key, values[kept & (values.map(len) > 1)])
        for key, values in [(semanticClass, classes), (sourceType, types)]
    ]
    for key, values in severals:
        joined = additional[values.index].map(lambda a: str(a.get(key, "")))
        combinations = joined.value_counts()
        l.log(f"{len(values)} synonyms have several values of {quote(key)}; they " \
            "count with the first one" + ("." if len(values) == 0 else ":"))
        # The 10 most frequent combinations.
        for value, count in combinations.head(10).items():
            l.log(f"- {quote(shorten(value, 60))}: {count}")
        if len(combinations) > 10:
            l.log(f"- and {len(combinations) - 10} other combinations.")

    # Synonyms that match the label of their concept, ignoring case.
    rowCount = len(data.index)
    l.log("Removing the synonyms that match their label...")
    terms   = data[idColumn].astype(str) + data[valueColumn].astype(str).str.lower()
    matches = synonymRows(data) & terms.isin(terms[data[attributeColumn] == labelClass])
    data    = data[~matches].reset_index(drop = True)
    l.log(f"{rowCount - len(data.index)} synonyms match their label.")
    l.log("Removing the synonyms that match their label completed.")

    l.printRowCount(data)
    printDataSummary(data)

    l.log("Writing the transformed HPO...")
    writeHugeCSV(data, outputFileTransformedFull)
    writeSettings(outputFileTransformedFull, transformSettings)
    l.log("Writing the transformed HPO completed.")

# ==============================================================================
# Embedding: the distance of every synonym to its label -> embed.csv and
# embed.development.csv, for SynTypeEval.py.
# ==============================================================================

def embedTerms(inputFile : str, outputFile : str, developmentSet : bool) -> None:
    """
    Embeds the synonyms of inputFile and their labels with every model of
    embedder.model_id and writes the normalised cosine distance of every
    synonym to its label to outputFile. Logs the mean and standard deviation
    (of the development set: the values for syntype.cosine_distance_mean and
    _std) and how well the distance separates expert from layperson synonyms.
    """
    from embedder import Embedder

    l.printHeader("Embedding the Development Set" if developmentSet else "Embedding the Evaluation Set")

    exitIfFileNotExist(inputFile)

    data = readTransformed(inputFile)

    l.log("Filtering the labels and synonyms...")
    # embed.csv keeps the format of the transformed HPO.
    distanceRows = synonymTable(data)
    types       = typesOf(distanceRows)
    synonyms    = distanceRows[valueColumn].astype(str).tolist()

    # The first label of every concept with synonyms (not those of parents and
    # children in the development set).
    labelRows   = data[(data[attributeColumn] == labelClass)
        & data[idColumn].isin(distanceRows[idColumn])].drop_duplicates(subset = idColumn)
    labels      = labelRows[valueColumn].astype(str).tolist()

    # Index of each synonym's label in labels, -1 if the concept has none.
    labelIndex  = distanceRows[idColumn].map(
        pd.Series(range(len(labels)), index = labelRows[idColumn])
    ).fillna(-1).astype(int).to_numpy()
    hasLabel    = labelIndex >= 0
    l.log(f"Found {len(labels)} labels and {len(synonyms)} synonyms " \
        f"({(~hasLabel).sum()} of a concept without label).")
    l.log("Filtering the labels and synonyms completed.")

    def describe(name : str, values : pd.Series) -> None:
        """
        Logs the mean, standard deviation, minimum and maximum of the non-NaN
        values, below a row naming them.
        """
        values = values.dropna()
        l.log(f"{name}:")
        l.log(f"  mean {values.mean():.4f}, std {values.std():.4f}.")
        l.log(f"  min {values.min():.4f}, max {values.max():.4f}.")

    def distance(a : np.ndarray, b : np.ndarray) -> np.ndarray:
        """
        Cosine distance of the rows of a and b: 1 minus their cosine
        similarity, from 0 (same direction) to 2 (opposite directions).
        """
        return 1 - np.sum(a * b, axis = 1) / (
            np.linalg.norm(a, axis = 1) * np.linalg.norm(b, axis = 1))

    l.log("Embedding the synonyms and labels with all embedding models...")
    embeddingModelsList = sorted(embeddingModels.keys())
    for index, embeddingModel in enumerate(embeddingModelsList):

        # One model at a time, so only one is held in memory.
        setUp = f"Setting up model {index + 1}/{len(embeddingModelsList)} {quote(embeddingModel)}"
        l.log(f"{setUp}...")
        configuration["embedder"]["model_id"] = [embeddingModels[embeddingModel]]
        embedder = Embedder(configuration)
        l.log(f"{setUp} completed.")

        l.log("Embedding the labels...")
        labelEmbeddings   = np.asarray(
            embedder.embed(labels)[embeddingModels[embeddingModel]])
        l.log("Embedding the labels completed.")
        l.log("Embedding the synonyms...")
        synonymEmbeddings = np.asarray(
            embedder.embed(synonyms)[embeddingModels[embeddingModel]])
        l.log("Embedding the synonyms completed.")

        a = synonymEmbeddings[hasLabel]
        b = labelEmbeddings[labelIndex[hasLabel]]

        m = "mean"
        s = "std"

        for distanceMetric in distanceMetrics:
            distanceColumn = distanceColumnPrefix.format(embeddingModel,
                distanceMetric)

            # NaN for synonyms without label.
            values           = np.full(len(synonyms), np.nan)
            values[hasLabel] = distance(a, b)

            # Normalised with syntype.cosine_distance_mean and _std, so every run
            # uses the same scale; with those of the data if they are None.
            values = pd.Series(values)
            mean   = values.mean() if cosineDistanceMean is None else cosineDistanceMean
            std    = values.std() if cosineDistanceStd is None else cosineDistanceStd
            normalised = (values - mean) / std
            distanceRows[distanceColumn] = normalised

            l.log(f"{embeddingModel}, {distanceMetric} distance to the label: " \
                f"{values.notna().sum()} synonyms ({values.isna().sum()} without label).")
            describe("All", values)
            for synonymType in synonymTypes:
                ofType = types == synonymType
                describe(f"Gold type {quote(synonymType)} ({ofType.sum()})", values[ofType])
            l.log(f"Normalised with mean {mean} and std {std}.")
            describe("Normalised, all", normalised)
            for synonymType in synonymTypes:
                ofType = types == synonymType
                describe(f"Normalised, gold type {quote(synonymType)}", normalised[ofType])

            if developmentSet:
                l.log("Set in config.yaml:")
                l.log(f"  syntype.cosine_distance_mean: {values.mean():.4f} (now {cosineDistanceMean}).")
                l.log(f"  syntype.cosine_distance_std: {values.std():.4f} (now {cosineDistanceStd}).")

            # A quick check how well the distances separate expert from
            # layperson synonyms: the strictly standardized mean difference.
            grouped = normalised[types.isin(synonymTypes)].groupby(
                types).agg([m, s]).reindex(synonymTypes)
            means = grouped[m].tolist()
            stds  = grouped[s].tolist()
            ssmd = (means[0] - means[1]) / math.sqrt(
                math.pow(stds[0], 2) + math.pow(stds[1], 2))
            l.log(f"Strictly standardized mean difference [{expertSynonymType} - " \
                f"{laypersonSynonymType}]: {ssmd:.4f}.")
            l.log(f"Difference of the means {means[0] - means[1]:.4f}, " \
                f"of the stds {stds[0] - stds[1]:.4f}.")


    l.log("Embedding the synonyms and labels with all embedding models completed.")

    writeHugeCSV(distanceRows, outputFile)
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

    if modelID not in configurationFile["llm"]["model_id"]:
        raise ValueError(f"{quote(modelID)} is not in llm.model_id.")
    modelIndex = configurationFile["llm"]["model_id"].index(modelID)

    modelName = modelNameOf(modelID)
    rawFile   = outputFileClass(modelName)

    l.printHeader(f"Classifying the Synonyms ({modelName})")

    exitIfFileNotExist(inputFileClass)

    gold     = readTransformed(inputFileClass)

    # The synonyms of the evaluation set, or of the development set ("test").
    synonyms = synonymTable(gold)

    def elements(attribute : str, key : str = idColumn,
            value : str = valueColumn) -> dict:
        """
        The values of an attribute's rows per key, in the order of the data.
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

    # The placeholder values of the prompts, one dictionary per synonym.
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
        for hpoID, synonym in zip(synonyms[idColumn], synonyms[valueColumn])
    ]

    # One (prompt column, answer column) pair per turn; the answer of the last
    # turn is the final answer.
    turns = [("{}{}".format(userRole, 1), answerColumn)]
    if chainOfThoughts:
        turns = [
            ("{}{}".format(userRole, 1), "{}{}".format(modelRole, 1)),
            ("{}{}".format(userRole, 2), "{}{}".format(modelRole, 2)),
            ("{}{}".format(userRole, 3), answerColumn)
        ]

    # The raw classification file, written after every turn.
    synonyms[systemColumn] = [modelName] * len(synonyms.index)
    for promptColumn, targetColumn in turns:
        synonyms[promptColumn] = [""] * len(synonyms.index)
        synonyms[targetColumn] = [""] * len(synonyms.index)

    settings = [name for name, value in experiment.model_dump().items() if value]
    l.log(f"Experiment {experimentIndex}: {applyFormat(settings)}.")
    # The raw and formatted files of the experiment, one per model.
    l.log(f"File(s): {quote(f'{filesPrefixApproachClass}_*.{csvFileFormat}')}.")
    l.log("Prompt file(s): " + applyFormat(
        [configurationFile["llm"]["prompt_files"][index] for index in promptIndexes]) + ".")
    goldTypes = typesOf(synonyms).value_counts()
    l.log(f"Classifying {len(synonyms.index)} synonyms of " \
        f"{synonyms[idColumn].nunique()} concepts; gold types: " + ", ".join(
            f"{goldType} {count}" for goldType, count in goldTypes.items()) + ".")

    # The answer store (checkpoint) and prompt log are named after the model
    # and the experiment (postfix), since start.sh starts many runs at once.
    l.log(f"Setting up the LLM ({modelName})...")
    with Model(configFile, modelIndex) as model:
        model.setPostfix(f"{testStr}_{configStr}_{chainOfThoughtsStr}_{fewShotStr}")
        l.log(f"Setting up the LLM ({modelName}) completed.")

        # Answers in the store are reused, so an interrupted run resumes; a
        # changed prompt has no stored answer, and a store written with other
        # model settings is refused.
        for turn, (promptColumn, targetColumn) in enumerate(turns):
            generating = f"Turn {turn + 1}/{len(turns)}: generating the answers"
            l.log(f"{generating}...")

            model.addPromptsFromFile(promptIndexes[turn], values)
            # The prompt of the first synonym, to see what the model is given.
            l.log(f"Prompt of turn {turn + 1} for the first synonym " \
                f"({quote(synonyms[valueColumn].iloc[0])}):\n{model.getPrompt(0, turn)}",
                cmdline = False)
            synonyms[promptColumn] = [model.getPrompt(index, turn) for index in range(len(values))]

            model.generate()

            synonyms[targetColumn] = [model.getAnswer(index, turn) for index in range(len(values))]
            writeHugeCSV(synonyms, rawFile)
            l.log(f"{generating} completed.")

        # The final answers are JSON objects (classification, confidence,
        # rationale).
        l.log("Parsing the answers...")
        parsed = [
            parseClassification(model.toJSON(index))
            for index in range(len(model.getMessageHistories()))
        ]
        synonyms[classificationKey]     = [classification for classification, _, _ in parsed]
        synonyms[confidenceColumn] = [confidence for _, confidence, _ in parsed]
        synonyms[rationaleColumn]  = [rationale for _, _, rationale in parsed]
        l.log(f"{sum(c != undefinedSynonymType for c in synonyms[classificationKey])} of " \
            f"{len(parsed)} answers contain a valid classification, " \
            f"{sum(r != '' for r in synonyms[rationaleColumn])} a rationale.")
        l.log("Parsing the answers completed.")
        writeHugeCSV(synonyms, rawFile)

        l.log("Writing the prompts and answers to the prompt log...")
        model.logPrompts()
        l.log("Writing the prompts and answers to the prompt log completed.")

    return synonyms

# ==============================================================================
# Formatting of the answers -> the formatted classification file (merged by
# SynTypeMerge.py).
# ==============================================================================

def formatAnswers(synonyms : pd.DataFrame, modelName : str) -> None:
    """
    Writes the formatted classification of a model (its answers parsed by
    classifySynonyms()), which SynTypeMerge.py merges, and logs every wrong or
    undefined classification.
    """
    l.printHeader(f"Formatting the Answers ({modelName})")

    classified = synonyms.copy()
    classified[answerColumn]     = classified[classificationKey].fillna(undefinedSynonymType)
    classified[confidenceColumn] = classified[confidenceColumn].fillna(-1).astype(int)
    classified                   = classified.drop(columns = classificationKey)

    writeHugeCSV(classified, outputFileClassFormatted(modelName))

    l.log("Logging the wrong classifications...")

    # The wrong classifications in the log help to improve the prompts.
    gold   = readTransformed(inputFileClass)
    labels = gold[gold[attributeColumn] == labelClass].copy().reset_index(drop = True)
    count  = 0

    l.log("Wrong or undefined classifications, as synonym (label): gold -> " \
        "classified (confidence),", cmdline = False)
    l.log("with the rationale in the next row.", cmdline = False)
    for (index, row), goldType in zip(classified.iterrows(), typesOf(classified)):
        # Undefined, or differing from a gold type the LLM classifies into.
        if (str(row[answerColumn]).lower() == undefinedSynonymType.lower() or
            (goldType in synonymTypes and
            str(row[answerColumn]).lower() != goldType)):
            # Shortened; the full texts are in the raw and formatted files.
            label = ", ".join(getElements(labels, row[idColumn], labelClass))
            l.log(f"{quote(shorten(row[valueColumn], 32))} " \
                f"(label {quote(shorten(label, 28))}): {goldType} -> " \
                f"{row[answerColumn]} ({row[confidenceColumn]}).", cmdline = False)
            l.log(f"    {quote(shorten(row[rationaleColumn], 90))}", cmdline = False)
            count = count + 1

    l.log(f"Wrong or undefined classifications: {count}.")
    l.log(f"Correct classifications: {len(classified.index) - count}.")
    l.log("Logging the wrong classifications completed.")

    logClassificationSummary(classified, modelName)


def logClassificationSummary(classified : pd.DataFrame, modelName : str) -> None:
    """
    Logs a summary of the formatted classification of a model, as
    SynTypeEval.py scores it: of the synonyms with an expert or layperson gold
    type, without those of the Few-Shot examples, a confusion table of gold
    against classified types, the mean confidence per classified type, and
    precision, recall and F1 per type (undefined answers count as wrong).
    """
    examples = [(example["id"], example["synonym"]) for example in fewShotExamples]
    scored = classified[
        typesOf(classified).isin(synonymTypes) & ~keysOf(classified).isin(examples)
    ]
    goldTypes = typesOf(scored)
    answers = synonymTypes + [undefinedSynonymType]

    l.log(f"Summary of {modelName}: {len(scored.index)} synonyms.")
    l.log(f"Gold types " + ", ".join(synonymTypes) + ", without the few-shot examples; " \
        f"{(scored[answerColumn] == undefinedSynonymType).sum()} answers undefined.")
    l.log(f"{'Gold / Classified':<20}" + "".join(f"{answer:>12}" for answer in answers))
    for goldType in synonymTypes:
        ofType = scored[goldTypes == goldType]
        l.log(f"{goldType:<20}" + "".join(
            f"{(ofType[answerColumn] == answer).sum():>12}" for answer in answers))

    for synonymType in synonymTypes:
        predicted = scored[answerColumn] == synonymType
        actual    = goldTypes == synonymType
        tp = (predicted & actual).sum()
        precision = tp / predicted.sum() if predicted.sum() > 0 else float("nan")
        recall    = tp / actual.sum() if actual.sum() > 0 else float("nan")
        f1        = 2 * tp / (predicted.sum() + actual.sum()) if predicted.sum() + actual.sum() > 0 else float("nan")
        confidence = scored.loc[predicted, confidenceColumn]
        l.log(f"{quote(synonymType)}: precision {precision:.3f}, recall {recall:.3f}, " \
            f"F1 {f1:.3f}.")
        l.log(f"{quote(synonymType)}: mean confidence {confidence.mean():.2f} " \
            f"of the {predicted.sum()} classified as {synonymType}.")

# ==============================================================================
# The steps of this run.
# ==============================================================================

def main() -> None:
    """
    Runs the steps: the preparation without an experiment index, otherwise
    the classification of the experiment.
    """
    # The log file is named after the start of the run only, so the command
    # line is its first line.
    l.log(f"Command: python3 {shlex.join(sys.argv)}")

    # Only the first of the runs started at the same time transforms the HPO; the
    # others wait for the lock. The reason is logged only once the lock is taken.
    if transformOutdated(log = False):
        with lockedFile(outputFileTransformedFull):
            if transformOutdated():
                with l.step("Transforming the HPO"):
                    transformHPO()
    else:
        l.log("The transformed HPO is up to date.")

    # Drawn once in the same way.
    if developmentSetOutdated(log = False):
        with lockedFile(outputFileTransformedDevelopment):
            if developmentSetOutdated():
                with l.step("Drawing the development and evaluation set"):
                    writeDevelopmentSet()
    else:
        l.log("The development and evaluation set are up to date.")

    if experimentIndex is None:
        # Preparation run: the distances of the development set, then of the
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
        # A failing model does not stop the others; each is unloaded from the GPU
        # before the next one.
        failed = []
        for index, modelID in enumerate(modelIDs):
            l.log(f"Model {index + 1}/{len(modelIDs)}: {quote(modelID)}.")
            modelName = modelNameOf(modelID)
            try:
                with l.step(f"Classifying the synonyms ({modelName})"):
                    synonyms = classifySynonyms(modelID)
                with l.step(f"Formatting the answers ({modelName})"):
                    formatAnswers(synonyms, modelName)
            except (Exception, SystemExit) as e:
                # The step's rows give the reason; the traceback goes to the
                # log file only.
                l.exception(f"Classifying with {quote(modelID)} failed.", cmdline = False)
                failed.append(modelID)

        if len(failed) > 0:
            l.error(f"{len(failed)} of {len(modelIDs)} model(s) failed: {applyFormat(failed)}.")
            sys.exit(1)


# vLLM starts its workers with "spawn", which imports this script again: the
# steps must only run in the main process.
if __name__ == "__main__":
    main()
