"""
Helpers shared by the scripts of syntype: the transformed HPO in the format of
the adapter package, reading and writing CSV files, parsing the answers of the
LLMs, progress bars and the data summary.
"""

import pandas       as pd
import numpy as np
import ast
import json
import sys
import os
import textwrap

from rich.table     import Table
from rich.console   import Console
from rich.progress  import Progress, BarColumn, TextColumn, TaskID
from rich.progress  import TaskProgressColumn, TimeElapsedColumn

# Prevent Python from generating .pyc bytecode files
sys.dont_write_bytecode = True

from config         import * 

from logger         import Logger, writeJSON

from adapter                    import BaseAdapterConfig, semanticClass, sourceType
from adapter                    import synonymClass as synonymAttribute
from adapter.BaseAdapterUtils   import additionalSeparator, isFile
from adapter.BaseAdapterUtils   import writeHugeCSV as writeHugeCSVAtomically

# The transformed HPO is kept in the format of the adapter package: one row
# per (id, attribute, value) with a dict of additional information, e.g. the
# synonym class and type of a synonym ({"semantic_class": "exact",
# "source_type": "layperson"}). The column names are those of the
# adapter.hpo section.
hpoAdapterConfig = BaseAdapterConfig.model_validate(configurationFile["adapter"]["hpo"])
idColumn         = hpoAdapterConfig.id_column
attributeColumn  = hpoAdapterConfig.attribute_column
valueColumn      = hpoAdapterConfig.value_column
additionalColumn = hpoAdapterConfig.additional_column

def additionalValues(additional, key : str) -> list:
    """
    The values of a key of the additional information of a row; a synonym
    listed several times for a concept has several, e.g. ["exact", "related"]
    (see the adapter's mergeAdditional()). [""] if the key is missing.
    """
    return str(additional.get(key, "")).split(additionalSeparator)

def synonymClassOf(additional) -> str:
    """
    The synonym class of a synonym (exact, related, ...): its first one.
    """
    return additionalValues(additional, semanticClass)[0]

def synonymTypeOf(additional) -> str:
    """
    The gold type of a synonym: its first source type, where a synonym without
    a source type (direct) is an expert synonym.
    """
    ret = additionalValues(additional, sourceType)[0]
    return expertSynonymType if ret == directSynonymType else ret

# ------------------------------------------------------------------------------
# The settings an output was made with are stored next to it
# (<output>.settings.json), so that it is made again when they change, not
# only when one of its input files is newer.
# ------------------------------------------------------------------------------

def isOutdated(output : str, inputs : list) -> bool:
    """
    Whether a file is missing or older than one of the files it is made from.
    """
    return not isFile(output) or any(
        os.path.getmtime(output) < os.path.getmtime(i) for i in inputs if isFile(i))

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
        if log: Logger().log(f"{os.path.basename(output)} is missing or older than " \
            f"{applyFormat([os.path.basename(i) for i in inputs])}.")
        return True

    changed = changedSettings(output, settings)
    if len(changed) > 0:
        if log: Logger().log(f"{os.path.basename(output)} was made with other or unrecorded " \
            f"settings: {applyFormat(changed)}.")
        return True

    return False

def readTransformed(file : str) -> pd.DataFrame:
    """
    Read a transformed HPO file, with the additional information as dicts.
    """
    ret = readCSV(file)
    # There are only few different dicts, so every one is parsed once.
    parsed = {entry : ast.literal_eval(entry) for entry in ret[additionalColumn].astype(str).unique()}
    ret[additionalColumn] = ret[additionalColumn].astype(str).map(parsed)
    return ret

def synonymRows(data : pd.DataFrame) -> pd.Series:
    """
    Whether a row of the transformed HPO is a synonym syntype classifies (one
    of synonymClasses).
    """
    return (data[attributeColumn] == synonymAttribute) & \
        data[additionalColumn].map(synonymClassOf).isin(synonymClasses)

def synonymTable(data : pd.DataFrame) -> pd.DataFrame:
    """
    The synonyms of the transformed HPO that syntype classifies, one row per
    synonym with its concept, synonym class, gold type and text, as in the
    classification files.
    """
    rows = data[synonymRows(data)]
    return pd.DataFrame({
        hpoidColumn   : rows[idColumn].tolist(),
        classColumn   : rows[additionalColumn].map(synonymClassOf).tolist(),
        typeColumn    : rows[additionalColumn].map(synonymTypeOf).tolist(),
        contentColumn : rows[valueColumn].astype(str).tolist(),
        systemColumn  : goldStandardSystem
    })

# The distance of every synonym to its label (SynTypeClassification.py,
# SynTypeEval.py): the cosine distance of every model of embedder.model_id.
# A model is named in the distance columns by syntype.embedding_model_names,
# or by its ID without the organisation.
cosineDistance  = "cosine"
distanceMetrics = [cosineDistance]

embeddingModels = {
    syntypeConfig.embedding_model_names.get(m, modelNameOf(m)) : m
    for m in configurationFile["embedder"]["model_id"]
}

def getRows(
    data: pd.DataFrame,
    hpoID: str,
    # Default to label attribute
    attribute=labelClass
) -> pd.DataFrame:
    """
    Return all rows of the transformed HPO matching a given HPO ID and
    attribute (a single attribute or several), or None.
    """
    # Filter by HPO ID
    ret = data[data[idColumn] == hpoID]

    if ret is not None and len(ret.index) > 0:
        if isinstance(attribute, str):
            ret = ret[ret[attributeColumn] == attribute]
        else:
            ret = ret[ret[attributeColumn].isin(attribute)]
    else:
        ret = None

    return ret


# Gets all values matching the given ID and attribute.
# Always returns a list (possibly empty).
def getElements(
    data: pd.DataFrame,
    hpoID: str,
    # Default to label attribute
    attribute=labelClass
) -> list:
    """
    Retrieve all values for a given HPO ID and attribute.

    If no matching rows exist, an empty list is returned.
    """
    ret = getRows(data, hpoID, attribute)

    # Extract the values if rows exist
    if ret is not None and len(ret.index) > 0:
        ret = ret[valueColumn].tolist()
    else:
        ret = []

    return ret


def newProgress() -> Progress:
    """
    Create a Rich progress bar with consistent formatting.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan"),
        TaskProgressColumn(),
        TextColumn("ET:"),
        TimeElapsedColumn(),
        TextColumn("Elem.s: {task.completed}/{task.total}"),
    )


def newTask(
    progress: Progress,
    iterations: int,
    text: str = "Taks"
) -> TaskID:
    """
    Add a new task to a Rich progress bar.
    """
    return progress.add_task(
        text.ljust(40),
        total=iterations
    )


def exitIfFileNotExist(file: str) -> None:
    """
    Terminate the program if the specified file does not exist.
    """
    if not isFile(file):
        Logger().error(f"{os.path.basename(file)} could not be found. Terminating.")
        sys.exit(1)


def readCSV(file: str = "") -> pd.DataFrame:
    """
    Read a CSV file from disk with logging.
    """
    Logger().printReadFileStart(file)
    ret = pd.read_csv(file, low_memory = False)
    Logger().printReadFileEnd(file)
    return ret


def writeHugeCSV(data: pd.DataFrame, file: str = "") -> None:
    """
    Write a DataFrame atomically as a comma-separated CSV file (see the
    adapter's writeHugeCSV()), creating its folder if needed.
    """
    folder = os.path.dirname(file)
    if folder:
        os.makedirs(folder, exist_ok = True)

    writeHugeCSVAtomically(data, file, ",")


def printDataSummary(data: pd.DataFrame = None) -> None:
    """
    Display a summary table showing absolute and relative
    frequencies of the attributes of the transformed HPO, the synonyms per
    synonym class.
    """
    console = Console()

    # Create summary table
    table = Table(title="Data Summary Report")

    table.add_column("Description")
    table.add_column("Abs")
    table.add_column("Rel")

    # Count occurrences per attribute (synonyms per synonym class)
    attributes   = data[attributeColumn].where(
        data[attributeColumn] != synonymAttribute,
        synonymAttribute + " (" + data[additionalColumn].map(synonymClassOf) + ")")
    value_counts = attributes.value_counts()

    for value, count in value_counts.items():
        relativeAmount = round(count * 100 / len(data.index))
        table.add_row(
            str(value),
            str(count).rjust(len(str(len(data.index)))),
            ("~" + str(relativeAmount) + "%").rjust(5)
        )

    # Render table to console
    console.print(table)


        

def parseClassification(answer) -> tuple[str, int, str]:
    """
    Classification, confidence and rationale of an answer parsed by
    model.toJSON(). The classification is (undefinedSynonymType, -1) if the
    answer is no JSON object with one of synonymTypes as classification and an
    integer confidence; the rationale is "" if the answer has none, also for
    an undefined classification.
    """
    classification, confidence = undefinedSynonymType, -1
    rationale = ""

    if isinstance(answer, dict):
        if answer.get(rationaleColumn) is not None:
            rationale = str(answer[rationaleColumn]).strip()

        if synonymClass in answer and confidenceColumn in answer:
            parsedClassification = str(answer[synonymClass]).strip().lower()
            try:
                parsedConfidence = int(answer[confidenceColumn])
            except (TypeError, ValueError):
                parsedConfidence = None

            if parsedClassification in synonymTypes and parsedConfidence is not None:
                classification, confidence = parsedClassification, parsedConfidence

    return classification, confidence, rationale

def quote(string : str) -> str:
    """
    The string in quotation marks, e.g. for the prompts and the log.
    """
    return quotationCharacter + string + quotationCharacter

def shorten(string, width : int = 60) -> str:
    """
    A text cut at a word to at most width characters, with " ..." where it
    is cut, so that a log row stays short (the full text is in the files).
    """
    return textwrap.shorten(str(string), width = width, placeholder = " ...")

def applyFormat(l : list[str] = []) -> str:
    """
    A list of strings as one text, each quoted and separated by commas, e.g.
    '"a", "b"'; a single string is only quoted, and an empty list gives
    "None".
    """
    string = "None"
    if len(l) > 0:
        if isinstance(l, str):
            string = quote(l)
        else:
            string = quote((quotationCharacter + ", " + 
                quotationCharacter).join(l))
    return string


# ------------------------------------------------------------------------------
# The development set (syntype.development_set): synonyms drawn at random from
# the transformed HPO with the information of their concepts, classified by
# runs with "test" (hpo.development.csv); and the evaluation set, the
# transformed HPO without these synonyms and without the synonyms of other
# gold types than expert and layperson, classified by the other runs
# (hpo.evaluation.csv).
# ------------------------------------------------------------------------------

# Both sets depend on the settings of the development set, on the gold types
# the evaluation set keeps and on the format of the transformed HPO.
developmentSetSettings = {
    "development_set" : syntypeConfig.development_set.model_dump(),
    "gold_types"      : synonymTypes,
    "format"          : [idColumn, attributeColumn, valueColumn, additionalColumn]
}

def developmentSetOutdated(log : bool = True) -> bool:
    """
    Whether the development or the evaluation set is missing, older than
    hpo.csv or drawn with other settings; the reason is logged if `log`.
    """
    return any([outdated(output, [outputFileTransformedFull], developmentSetSettings, log)
        for output in [outputFileTransformedDevelopment, outputFileTransformedEvaluation]])

def logSynonymCounts(name : str, data : pd.DataFrame) -> None:
    """
    Logs how many synonyms of every synonym class and gold type a transformed
    HPO file holds, those syntype classifies (synonymTable()).
    """
    l = Logger()
    synonyms = synonymTable(data)
    types    = synonyms[typeColumn].replace("", "(none)")
    total    = types.value_counts()
    counts   = pd.crosstab(synonyms[classColumn], types)[total.index]
    l.log(f"{name}: {len(synonyms.index)} synonyms: " + ", ".join(
        f"{t} {n}" for t, n in total.items()) + ".")
    for synonymClass, row in counts.iterrows():
        l.log(f"{name}, {synonymClass}: {row.sum()} synonyms: " + ", ".join(
            f"{t} {n}" for t, n in row.items() if n > 0) + ".")

def writeDevelopmentSet() -> pd.DataFrame:
    """
    Draws syntype.development_set.size_per_type synonyms of every gold type
    (expert, layperson) at random from hpo.csv and writes them to
    hpo.development.csv, in the adapter's format, with the information of their
    concepts: label, definition, comment, the child rows of the concepts (to
    their children and from their parents) and the labels of those parents
    and children, but no other synonym. Writes hpo.csv without these synonyms
    and without the synonyms of other gold types than expert and layperson to
    hpo.evaluation.csv, so that the development synonyms are never
    evaluated. Records the settings next to both.

    :return: The development set.
    """
    l = Logger()
    settings = syntypeConfig.development_set

    exitIfFileNotExist(outputFileTransformedFull)
    data  = readTransformed(outputFileTransformedFull)
    types = data[additionalColumn].map(synonymTypeOf)
    synonyms = synonymRows(data)

    # The synonyms of every gold type, drawn with the configured seed.
    selected = []
    for synonymType in synonymTypes:
        candidates = data[synonyms & (types == synonymType)]
        if len(candidates.index) < settings.size_per_type:
            l.error(f"Only {len(candidates.index)} synonyms of gold type " \
                f"{quote(synonymType)}, fewer than syntype.development_set." \
                f"size_per_type ({settings.size_per_type}).")
            sys.exit(1)
        selected.append(candidates.sample(n = settings.size_per_type,
            random_state = settings.seed).index)
        l.log(f"Drawing {settings.size_per_type} of {len(candidates.index)} " \
            f"synonyms of gold type {quote(synonymType)} (seed {settings.seed}).")
    selectedRows = data.index.isin(np.concatenate(selected))

    # The information of their concepts, and the labels of their parents and
    # children (a child row reads "id has child value").
    concepts  = set(data.loc[selectedRows, idColumn])
    attribute = data[attributeColumn]
    conceptRows  = data[idColumn].isin(concepts) & attribute.isin(
        [labelClass, definitionClass, commentClass])
    relationRows = (attribute == childrenClass) & (
        data[idColumn].isin(concepts) | data[valueColumn].isin(concepts))
    related   = (set(data.loc[relationRows, idColumn]) |
        set(data.loc[relationRows, valueColumn])) - concepts
    relatedLabelRows = data[idColumn].isin(related) & (attribute == labelClass)

    result = data[selectedRows | conceptRows | relationRows | relatedLabelRows].reset_index(drop = True)
    l.log(f"Development set: {selectedRows.sum()} synonyms of {len(concepts)} concepts, " \
        f"{len(result.index)} rows.")
    l.log(f"Development set: {relationRows.sum()} relations, {len(related)} labels " \
        f"of parents and children.")
    printDataSummary(result)
    logSynonymCounts("Development set", result)

    # The evaluation set: every row of hpo.csv except the drawn synonyms and
    # the synonyms of other gold types than expert and layperson (e.g.
    # abbreviation, plural; a synonym listed several times counts with its
    # first type). A drawn synonym listed several times (with other classes
    # or types) is removed with all its rows.
    drawn       = set(zip(data.loc[selectedRows, idColumn], data.loc[selectedRows, valueColumn]))
    drawnRows   = synonyms & pd.Series(list(zip(data[idColumn], data[valueColumn])),
        index = data.index).isin(drawn)
    otherRows   = (data[attributeColumn] == synonymAttribute) & ~types.isin(synonymTypes)
    evaluation  = data[~(drawnRows | otherRows)].reset_index(drop = True)
    l.log(f"Evaluation set: {len(evaluation.index)} of the {len(data.index)} rows of " \
        f"{os.path.basename(outputFileTransformedFull)}.")
    l.log(f"Left out: {drawnRows.sum()} rows of the drawn synonyms, " \
        f"{(otherRows & ~drawnRows).sum()} of other gold types.")
    logSynonymCounts("Evaluation set", evaluation)

    writeHugeCSV(result, outputFileTransformedDevelopment)
    writeSettings(outputFileTransformedDevelopment, developmentSetSettings)
    writeHugeCSV(evaluation, outputFileTransformedEvaluation)
    writeSettings(outputFileTransformedEvaluation, developmentSetSettings)
    return result
