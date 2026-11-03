import pandas       as pd
import numpy as np
import ast
import sys
import os

from rich.table     import Table
from rich.console   import Console
from rich.progress  import Progress, BarColumn, TextColumn, TaskID
from rich.progress  import TaskProgressColumn, TimeElapsedColumn

# Prevent Python from generating .pyc bytecode files
sys.dont_write_bytecode = True

from config         import * 

from logger         import Logger

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

# The similarity of every synonym to its label (syntype.py, syntypeeval.py):
# the cosine similarity of every model of embedder.model_id. A model is named
# in the similarity columns by syntype.embedding_model_names, or by its ID
# without the organisation.
cosineSimilarity  = "cosine"
similarityMetrics = [cosineSimilarity]

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


        

def parseClassification(answer) -> tuple[str, int]:
    """
    Classification and confidence of an answer parsed by model.toJSON(), or
    (undefinedSynonymType, -1) if the answer is no JSON object with one of
    synonymTypes as classification and an integer confidence.
    """
    ret = (undefinedSynonymType, -1)

    if (isinstance(answer, dict) and synonymClass in answer and
        confidenceColumn in answer):
        classification = str(answer[synonymClass]).strip().lower()
        try:
            confidence = int(answer[confidenceColumn])
        except (TypeError, ValueError):
            confidence = None

        if classification in synonymTypes and confidence is not None:
            ret = (classification, confidence)

    return ret

def getHPOIDs(data : pd.DataFrame) -> list:
    ret = []

    if reduceToTestIDs:
        ret = testIDs
    else:
        if data is not None and len(data.index) > 0 and hpoidColumn in data.columns:
            ret = list(set(data[hpoidColumn].tolist()))

    return ret

    


def quote(string : str) -> str:
    return quotationCharacter + string + quotationCharacter

def applyFormat(l : list[str] = []) -> str:
    string = "None"
    if len(l) > 0:
        if isinstance(l, str):
            string = quote(l)
        else:
            string = quote((quotationCharacter + ", " + 
                quotationCharacter).join(l))
    return string