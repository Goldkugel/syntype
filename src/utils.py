import pandas       as pd
import numpy as np
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

def isFile(file: str = "") -> bool:
    """
    Check whether a given path exists and is a file.
    """
    return os.path.isfile(file)


def getRows(
    data: pd.DataFrame,
    hpoID: str,
    # Default to label class
    className=labelClass
) -> pd.DataFrame:
    """
    Return all rows matching a given HPO ID and class name.

    Supports:
    - Single class name (string)
    - Multiple class names (list or iterable)
    """
    # Filter by HPO ID
    ret = data[data[hpoidColumn] == hpoID]

    if ret is not None and len(ret.index) > 0:
        if isinstance(className, str):
            # Filter by a single class
            ret = ret[ret[classColumn] == className]
        else:
            # Filter by multiple classes
            ret = ret[ret[classColumn].isin(className)]
    else:
        ret = None

    return ret


# Gets all elements matching the given ID and class.
# Always returns a list (possibly empty).
def getElements(
    data: pd.DataFrame,
    hpoID: str,
    # Default to label class
    className=labelClass
) -> list:
    """
    Retrieve all content values for a given HPO ID and class.

    If no matching rows exist, an empty list is returned.
    """
    ret = getRows(data, hpoID, className)

    # Extract content values if rows exist
    if ret is not None and len(ret.index) > 0:
        ret = ret[contentColumn].tolist()
    else:
        ret = []

    return ret


def newProgress() -> Progress:
    """
    Create a Rich progress bar with consistent formatting.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=progressBarColor),
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
        text.ljust(progressBarTextLength),
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
    l = Logger()
    tmpfile = file + ".tmp"
    l.log("Writing in temporary file...")
    writeCSV(data, tmpfile)
    l.log("Writing in temporary file completed. Replacing old data with new data...")
    os.replace(tmpfile, file)
    l.log("Old data replaced with new data.")


def writeCSV(data: pd.DataFrame, file: str = "") -> None:
    """
    Write a DataFrame to disk as a CSV file with logging.
    """
    Logger().printWriteFileStart(file)

    folder = os.path.dirname(file)
    if folder:
        os.makedirs(folder, exist_ok = True)

    with open(file, "w", newline="") as f:
        data.to_csv(f, index = False)
        f.flush()
        
    Logger().printWriteFileEnd(file)


def printDataSummary(data: pd.DataFrame = None) -> None:
    """
    Display a summary table showing absolute and relative
    frequencies of values in the class column.
    """
    console = Console()

    # Create summary table
    table = Table(title="Data Summary Report")

    table.add_column("Description")
    table.add_column("Abs")
    table.add_column("Rel")

    # Count occurrences per class
    value_counts = data[classColumn].value_counts()

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