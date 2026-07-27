import pandas       as pd
import Levenshtein  as lev
import numpy as np
import sys
import os
import json

from datetime       import datetime as dt
from rich.table     import Table
from rich.console   import Console
from rich.progress  import Progress, BarColumn, TextColumn, TaskID
from rich.progress  import TaskProgressColumn, TimeElapsedColumn
from owlready2      import *
from rdflib         import Namespace, RDF, Literal

# Prevent Python from generating .pyc bytecode files
sys.dont_write_bytecode = True

from config         import * 

# Common ontology namespaces used for RDF / OWL processing
OBO      = Namespace("http://purl.obolibrary.org/obo/")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
OWL      = Namespace("http://www.w3.org/2002/07/owl#")
RDF      = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")


def createDir(dir: str = "") -> bool:
    """
    Create a directory under the configured data directory.
    Does nothing if the directory already exists.
    """
    os.makedirs(os.path.join(dataDir, dir), exist_ok = True)


def isFile(file: str = "") -> bool:
    """
    Check whether a given path exists and is a file.
    """
    return os.path.isfile(file)


def replaceQuotes(
    text: str = "",
    quotationChar: str = quotationCharacter
) -> str:
    """
    Normalize different quotation characters into a single configured character.
    Useful for text normalization and downstream string comparisons.
    """
    return (
        text.replace("'", quotationChar)
            .replace("’", quotationChar)
            .replace("‘", quotationChar)
            .replace("\"", quotationChar)
    )


# Retrieve labels of child concepts for a given HPO ID
def getChildLabels(
    data: pd.DataFrame,
    hpoID: str
) -> list:
    """
    Get all label values of child concepts for the given HPO ID.

    This first retrieves all child IDs, then resolves their labels.
    """
    ret = []

    # Get IDs of child concepts
    childIDs = getElements(data, hpoID, childrenClass)

    # Retrieve labels for each child ID
    for childID in childIDs:
        ret += getElements(data, childID, labelClass)

    return ret


# Retrieve labels of parent concepts for a given HPO ID
def getParentLabels(
    data: pd.DataFrame,
    hpoID: str
) -> list:
    """
    Get all label values of parent concepts for the given HPO ID.
    """
    ret = []

    # Filter rows where the current concept appears as content
    contentFilter = data[data[contentColumn] == hpoID]

    # Restrict to rows describing parent-child relationships
    classFilter = contentFilter[
        contentFilter[classColumn] == childrenClass
    ]

    # Extract parent IDs
    parentIDs = classFilter[hpoidColumn].tolist()

    # Retrieve labels for each parent ID
    for parentID in parentIDs:
        ret += getElements(data, parentID, labelClass)

    return ret


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


def isFile(file: str = "") -> bool:
    """
    Check whether a given path exists and is a file.
    (Duplicate helper; mirrors earlier definition.)
    """
    return os.path.isfile(file)


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
        log(f"{os.path.basename(file)} could not be found. Terminating.")
        sys.exit(1)


def printHeader(text: str = "") -> None:
    """
    Print a formatted header block to the log/console.
    """
    log(headerSeparator)
    log(headerChar + text.center(headerLen - 2) + headerChar)
    log(headerSeparator)


def printProcessing(file: str = "") -> None:
    """
    Log the start of file processing.
    """
    log("Processing file '" + os.path.basename(file) + "'.")


def printProcessingDone(file: str = "") -> None:
    """
    Log completion of file processing.
    """
    log("Processing file '" + os.path.basename(file) + "' completed.")


def printRowCount(data: pd.DataFrame = None) -> None:
    """
    Log the number of rows in a DataFrame.
    """
    log("Row count: " + str(len(data.index)))


def printRead(file: str = "") -> None:
    """
    Log file read start.
    """
    log("Reading file '" + os.path.basename(file) + "'.")


def printWrite(file: str = "") -> None:
    """
    Log file write start.
    """
    log("Writing file '" + os.path.basename(file) + "'.")


def printReadDone(file: str = "") -> None:
    """
    Log file read completion.
    """
    log("Reading file '" + os.path.basename(file) + "' completed.")


def printWriteDone(file: str = "") -> None:
    """
    Log file write completion.
    """
    log("Writing file '" + os.path.basename(file) + "' completed.")


def readPickle(file: str = "") -> pd.DataFrame:
    """
    Read a pickled DataFrame from disk with logging.
    """
    printRead(file)
    ret = pd.read_pickle(file)
    printReadDone(file)
    return ret

def readCSV(file: str = "") -> pd.DataFrame:
    """
    Read a CSV file from disk with logging.
    """
    printRead(file)
    ret = pd.read_csv(file, low_memory = False)
    printReadDone(file)
    return ret

def log(
    string: str,
    cmdline: bool = True,
    file: str = logFile
) -> None:
    """
    Log a timestamped message to file and optionally to stdout.
    """
    if string is not None:
        myfile = open(file=str(file), mode="a")

        # Prefix log message with timestamp
        string = (
            "[" + dt.now().strftime("%Y-%m-%d %H:%M:%S") + "] "
            + string
        )

        if cmdline:
            print(string)

        myfile.write(string + "\n")


def writePickle(data: pd.DataFrame, file: str = "") -> None:
    """
    Write a DataFrame to disk as a pickle with logging.
    """
    printWrite(file)
    pd.to_pickle(data, file)
    printWriteDone(file)

def writeHugeCSV(data: pd.DataFrame, file: str = "") -> None:
    tmpfile = file + ".tmp"
    log("Writing in temporary file...")
    writeCSV(data, tmpfile)
    log("Writing in temporary file completed. Replacing old data with new data...")
    os.replace(tmpfile, file)
    log("Old data replaced with new data.")


def writeCSV(data: pd.DataFrame, file: str = "") -> None:
    """
    Write a DataFrame to disk as a CSV file with logging.
    """
    printWrite(file)

    with open(file, "w", newline="") as f:
        data.to_csv(f, index = False)
        f.flush()
        
    printWriteDone(file)


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


def removeEmptyRows(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows where the content column is empty.

    Returns:
        - A cleaned DataFrame with empty rows removed
        - The number of rows that were removed
    """
    # Work on a copy to avoid mutating the original DataFrame
    ret = data.copy()

    remove = []

    # Identify rows where the content column is empty or missing
    for index, row in ret.iterrows():
        if len(str(row[contentColumn])) == 0:
            remove.append(index)

    # Drop all empty rows at once
    ret = ret.drop(index=remove)

    # Reset index to keep it continuous after row removal
    ret = ret.reset_index(drop=True)

    return ret, len(remove)


def getConceptID(string: str = "") -> str:
    """
    Extract a concept identifier from a URI or path-like string.

    Example:
        'http://purl.obolibrary.org/obo/HP_0000118'
        -> 'HP:0000118'
    """
    ret = ""

    if string is not None and "/" in string:
        # Extract the last path segment
        parts = string.split("/")
        ret = parts[len(parts) - 1]

        # Convert underscore-based IDs to colon-based IDs
        # (e.g. HP_0000118 -> HP:0000118)
        if "_" in ret:
            ret = ret.replace("_", ":")

    return ret


def getSynonymTypeFromString(string: str = "") -> str:
    """
    Determine the synonym *type* based on markers found in the source string.

    The first matching type found will be returned.
    """
    ret = ""

    if string is not None:
        # Layperson-friendly synonym
        if owlSourceSynonymTypeLayperson in string:
            ret = laypersonSynonymType

        # Abbreviation synonym
        if owlSourceSynonymTypeAbbreviation in string:
            ret = abbreviationSynonymType

        # Obsolete synonym
        if owlSourceSynonymTypeObsolete in string:
            ret = obsoleteSynonymType

        # UK spelling variant
        if owlSourceSynonymTypeUKSpelling in string:
            ret = ukSpellingSynonymType

        # Plural form variant
        if owlSourceSynonymTypePlural in string:
            ret = pluralFormSynonymType

        # Allelic requirement synonym
        if owlSourceSynonymTypeAllelic in string:
            ret = allelicRequirementSynonymType

        # Direct synonym (no qualifier)
        if directSynonymType in string:
            ret = directSynonymType

    return ret


def getSynonymClassFromString(string: str = "") -> str:
    """
    Determine the synonym *class* (exact, related, broad, narrow)
    based on ontology source markers in the string.
    """
    ret = ""

    if string is not None:
        # Exact synonym
        if owlSourceExactSynonym in string:
            ret = exactSynonymClass

        # Related synonym
        if owlSourceRelatedSynonym in string:
            ret = relatedSynonymClass

        # Broad synonym
        if owlSourceBoradSynonym in string:
            ret = broadSynonymClass

        # Narrow synonym
        if owlSourceNarrowSynonym in string:
            ret = narrowSynonymClass

    return ret


def getSynonymsAndTypes(hpo: Ontology = None) -> pd.DataFrame:
    """
    Extract all synonyms for HPO concepts, including:
    - Direct synonym annotations
    - Axiom-annotated synonyms with explicit synonym types

    Returns a normalized DataFrame suitable for downstream processing.
    """
    ret = None

    if hpo is not None:
        # Convert Owlready2 ontology into an RDFLib graph
        g = hpo.world.as_rdflib_graph()

        # SPARQL query retrieves:
        # - Concept ID
        # - Synonym text
        # - Synonym class (exact, broad, narrow, related)
        # - Optional synonym type (e.g. layperson, abbreviation)
        query = """
        SELECT ?hpoID ?synonym ?synclass ?syntype WHERE 
        {
            {
                # Direct class annotations without axiom metadata
                ?hpoID ?synclass ?synonym .
                FILTER(?synclass IN (
                    oboInOwl:hasExactSynonym,
                    oboInOwl:hasBroadSynonym,
                    oboInOwl:hasNarrowSynonym,
                    oboInOwl:hasRelatedSynonym
                ))
                # Direct annotations have no synonym type metadata
                BIND(\"""" + directSynonymType + """\" AS ?syntype)
            }
            UNION
            {
                # Synonyms defined via OWL axioms (allow extra annotations)
                ?axiom rdf:type owl:Axiom .
                ?axiom owl:annotatedSource ?hpoID .
                ?axiom owl:annotatedProperty ?synclass .
                ?axiom owl:annotatedTarget ?synonym .
                FILTER(?synclass IN (
                    oboInOwl:hasExactSynonym,
                    oboInOwl:hasBroadSynonym,
                    oboInOwl:hasNarrowSynonym,
                    oboInOwl:hasRelatedSynonym
                ))
                # Optional synonym type (e.g. layperson, obsolete)
                OPTIONAL { ?axiom oboInOwl:hasSynonymType ?syntype }
            }
        }
        """

        # Execute query with required namespace bindings
        result = g.query(
            query,
            initNs={
                "rdf": RDF,
                "owl": OWL,
                "obo": OBO,
                "oboInOwl": OBOINOWL,
            }
        )
        
        # Accumulate query results into Python lists
        hpoIDs          = []
        synonyms        = []
        synonymClasses  = []
        synonymTypes    = []

        for row in result:
            # Normalize HPO ID format
            hpoIDs.append(getConceptID(str(row.hpoID)))

            # Raw synonym literal
            synonyms.append(row.synonym)

            # Map RDF property to internal synonym class
            synonymClasses.append(
                getSynonymClassFromString(str(row.synclass))
            )

            # Map OWL synonym type annotation to internal type
            synonymTypes.append(
                getSynonymTypeFromString(str(row.syntype))
            )

        # Build a standardized DataFrame representation
        ret = pd.DataFrame({
            hpoidColumn   : hpoIDs,
            classColumn   : synonymClasses,
            typeColumn    : synonymTypes,
            contentColumn : synonyms,
            systemColumn  : [goldStandardSystem] * len(hpoIDs),
            roundColumn   : [-1] * len(hpoIDs)
        })

    return ret


def getComments(hpo: Ontology = None) -> pd.DataFrame:
    """
    Extract rdfs:comment annotations for HPO concepts.

    Includes:
    - Direct comment annotations
    - Axiom-based comment annotations
    """
    ret = None

    if hpo is not None:
        g = hpo.world.as_rdflib_graph()

        # SPARQL query retrieves comments from both direct and axiom-based sources
        query = """
            SELECT DISTINCT ?hpoID ?comment WHERE {
                {
                    # Direct comment annotation
                    ?hpoID rdfs:comment ?comment .
                }
                UNION
                {
                    # Axiom-based comment annotation
                    ?axiom rdf:type owl:Axiom .
                    ?axiom owl:annotatedSource ?hpoID .
                    ?axiom owl:annotatedProperty rdfs:comment .
                    ?axiom owl:annotatedAnnotatedTarget ?comment .
                }
            }
        """

        result = g.query(
            query,
            initNs={
                "rdf": RDF,
                "owl": OWL,
                "obo": OBO,
                "oboInOwl": OBOINOWL,
            }
        )

        hpoIDs   = []
        comments = []

        for row in result:
            hpoIDs.append(getConceptID(str(row.hpoID)))
            comments.append(row.comment)

        # Normalize comments into standard DataFrame format
        ret = pd.DataFrame({
            hpoidColumn   : hpoIDs,
            classColumn   : [commentClass] * len(hpoIDs),
            typeColumn    : [""] * len(hpoIDs),
            contentColumn : comments,
            systemColumn  : goldStandardSystem,
            roundColumn   : [-1] * len(hpoIDs)
        })

    return ret


def getDefinitions(hpo: Ontology = None) -> pd.DataFrame:
    """
    Extract textual definitions for HPO concepts
    (obo:IAO_0000115 annotations).
    """
    ret = None

    if hpo is not None:
        g = hpo.world.as_rdflib_graph()

        # Retrieve definitions from direct and axiom-based annotations
        query = """
            SELECT DISTINCT ?hpoID ?definition WHERE {
                {
                    ?hpoID obo:IAO_0000115 ?definition .
                }
                UNION
                {
                    ?axiom rdf:type owl:Axiom .
                    ?axiom owl:annotatedProperty obo:IAO_0000115 .
                    ?axiom owl:annotatedTarget ?definition .
                    ?axiom owl:annotatedSource ?hpoID .
                }
            }
        """

        result = g.query(
            query,
            initNs={
                "rdf": RDF,
                "owl": OWL,
                "obo": OBO,
                "oboInOwl": OBOINOWL,
            }
        )

        hpoIDs      = []
        definitions = []

        for row in result:
            hpoIDs.append(getConceptID(str(row.hpoID)))
            definitions.append(row.definition)

        # Normalize definitions into standard DataFrame format
        ret = pd.DataFrame({
            hpoidColumn   : hpoIDs,
            classColumn   : [definitionClass] * len(hpoIDs),
            typeColumn    : [""] * len(hpoIDs),
            contentColumn : definitions,
            systemColumn  : goldStandardSystem,
            roundColumn   : [-1] * len(hpoIDs)
        })

    return ret


def getLabels(hpo: Ontology = None) -> pd.DataFrame:
    """
    Extract rdfs:label annotations for HPO concepts.
    """
    ret = None

    if hpo is not None:
        g = hpo.world.as_rdflib_graph()

        # Simple query for concept labels
        query = """
            SELECT ?hpoID ?label WHERE {
                ?hpoID rdfs:label ?label .
            }
        """

        result = g.query(
            query,
            initNs={
                "rdf": RDF,
                "owl": OWL,
                "obo": OBO,
                "oboInOwl": OBOINOWL,
            }
        )

        hpoIDs = []
        labels = []

        for row in result:
            hpoIDs.append(getConceptID(str(row.hpoID)))
            labels.append(row.label)

        # Normalize labels into standard DataFrame format
        ret = pd.DataFrame({
            hpoidColumn   : hpoIDs,
            classColumn   : [labelClass] * len(hpoIDs),
            typeColumn    : [""] * len(hpoIDs),
            contentColumn : labels,
            systemColumn  : goldStandardSystem,
            roundColumn   : [-1] * len(hpoIDs)
        })

    return ret


def getChildren(hpo: Ontology = None) -> pd.DataFrame:
    """
    Extract parent–child (subClassOf) relationships from the ontology.

    Each row represents:
        parent HPO ID -> child HPO ID
    """
    ret = None

    if hpo is not None:
        # Convert Owlready2 ontology into an RDFLib graph
        g = hpo.world.as_rdflib_graph()

        # Retrieve all subclass relationships
        query = """
            SELECT ?child ?parent WHERE {
                ?child rdfs:subClassOf ?parent .
            }
        """

        result = g.query(
            query,
            initNs={
                "rdf": RDF,
                "owl": OWL,
                "obo": OBO,
                "oboInOwl": OBOINOWL,
            }
        )

        children = []
        parents  = []

        for row in result:
            # Normalize HPO identifiers
            children.append(getConceptID(str(row.child)))
            parents.append(getConceptID(str(row.parent)))

        # Normalize relationships into standard DataFrame format
        ret = pd.DataFrame({
            hpoidColumn   : parents,
            classColumn   : [childrenClass] * len(parents),
            typeColumn    : [""] * len(parents),
            contentColumn : children,
            systemColumn  : goldStandardSystem,
            roundColumn   : [-1] * len(parents)
        })

    return ret


def getReferences(hpo: Ontology = None) -> pd.DataFrame:
    """
    Extract database cross-references (DbXrefs) for HPO concepts.

    Includes:
    - Direct hasDbXref annotations
    - Axiom-annotated DbXrefs
    """
    ret = None

    if hpo is not None:
        g = hpo.world.as_rdflib_graph()

        # Retrieve cross-references from both direct and axiom-based annotations
        query = """
            SELECT DISTINCT ?hpoID ?xref WHERE {
                {
                    ?hpoID oboInOwl:hasDbXref ?xref .
                }
                UNION
                {
                    ?axiom rdf:type owl:Axiom .
                    ?axiom owl:annotatedSource ?hpoID .
                    ?axiom oboInOwl:hasDbXref ?xref .
                }
            }
        """

        result = g.query(
            query,
            initNs={
                "rdf": RDF,
                "owl": OWL,
                "obo": OBO,
                "oboInOwl": OBOINOWL,
            }
        )

        hpoIDs = []
        xrefs  = []

        for row in result:
            hpoIDs.append(getConceptID(str(row.hpoID)))
            xrefs.append(row.xref)

        # Normalize references into standard DataFrame format
        ret = pd.DataFrame({
            hpoidColumn   : hpoIDs,
            classColumn   : [referenceClass] * len(hpoIDs),
            typeColumn    : [""] * len(hpoIDs),
            contentColumn : xrefs,
            systemColumn  : goldStandardSystem,
            roundColumn   : [-1] * len(hpoIDs)
        })

    return ret


def checkEntries(data: pd.DataFrame, hpoID: str = "") -> pd.DataFrame:
    """
    Validate and deduplicate entries for a single HPO concept.

    Special handling is applied for synonyms:
    - Prefer axiom-annotated synonyms over direct ones
    - Remove synonyms that duplicate the concept label
    """
    ret = None

    if data is not None and hpoID is not None:
        # Work on a copy of all entries for this concept
        subset = data[data[hpoidColumn] == hpoID].copy()

        if len(subset.index) > 0:
            # Check for duplicated content values
            content = list(set(subset[contentColumn]))
            if len(content) != len(subset.index):

                # Only synonyms may legitimately appear more than once
                # Prefer axiom-based synonyms over direct synonyms
                axiomSynonyms = list(
                    subset.loc[
                        (subset[classColumn].isin(synonymClasses)) &
                        (subset[typeColumn] != directSynonymType),
                        contentColumn
                    ]
                )

                # Remove direct synonyms when an axiom-based synonym exists
                subset.drop(
                    index=subset[
                        (subset[classColumn].isin(synonymClasses)) &
                        (subset[typeColumn] == directSynonymType) &
                        (subset[contentColumn].isin(axiomSynonyms))
                    ].index,
                    inplace=True
                )

                # Synonyms may sometimes duplicate the concept label
                # In that case, drop the synonym entry
                labels = getElements(subset, hpoID, labelClass)
                subset.drop(
                    index=subset[
                        (subset[contentColumn].isin(labels)) &
                        (subset[classColumn].isin(synonymClasses))
                    ].index,
                    inplace=True
                )

            ret = subset

    return ret

def formatAnswerGeneration(txt : str, label : str) -> list:
    ret = None

    # Retrieve and normalize raw LLM answer text
    answer = str(txt).strip()
    answer = answer.replace("```json", "")
    answer = answer.replace("```", "")
    answer = answer.replace("\n", "")
    answer = answer.replace("'", '"')

    # Attempt to isolate a JSON object in the response
    if "{" in answer and "}" in answer:
        answer = answer[answer.index("{"):answer.index("}") + 1]

        try:
            # Parse JSON content
            jsonAnswer = json.loads(answer)

            if jsonAnswer is not None:
                # Expect a dictionary containing "exact_synonyms"
                if isinstance(jsonAnswer, dict) and "exact_synonyms" in \
                        dict(jsonAnswer).keys():
                    
                    ret = jsonAnswer["exact_synonyms"]

                    # Validate synonym list structure
                    if (
                        ret is not None
                        and isinstance(ret, list)
                        and all(isinstance(item, str) for item in ret)
                    ):
                        # Remove duplicates and empty strings
                        ret = list(set(ret))
                        if "" in ret:
                            ret.remove("")

                        # Remove label if it appears among synonyms
                        
                        if label in ret:
                            ret.remove(label)
                    else:
                        ret = None
                else:
                    ret = None
            else:
                ret = None
        except json.JSONDecodeError:
            # JSON parsing failed
            ret = None
    else:
        # No JSON-like structure found
        ret = None
        
    return ret
        

def formatAnswerClassification(txt : str) -> tuple[str, int]:
    ret = undefinedSynonymType
    confidence = -1

    if txt is not None and "{" in txt and "}" in txt:
        start   = txt.rfind("{")
        end     = txt.rfind("}")
        if start < end:
            try:
                plain = txt[start:end+1]
                while "\"\"" in plain:
                    plain = plain.replace("\"\"", "\"")
                jsonAnswer = json.loads(plain)
                if isinstance(jsonAnswer, dict):
                    j = dict(jsonAnswer)
                    if synonymClass in j.keys() and confidenceColumn in j.keys():
                        a = str(j[synonymClass]).lower()
                        if a in synonymClasses:
                            confidence = int(j[confidenceColumn])
                            ret = a
            except:
                confidence = -1

    return ret, confidence

def formatAnswerClassificationType(txt : str) -> tuple[str, int]:
    ret = undefinedSynonymType
    confidence = -1

    if txt is not None:
        start   = txt.rfind("{")
        end     = txt.rfind("}")
        if start < end:
            try:
                plain = txt[start:end+1]
                while "\"\"" in plain:
                    plain = plain.replace("\"\"", "\"")
                jsonAnswer = json.loads(plain)
                if isinstance(jsonAnswer, dict):
                    j = dict(jsonAnswer)
                    if synonymClass in j.keys() and confidenceColumn in j.keys():
                        a = str(j[synonymClass]).lower()
                        if a in synonymTypes:
                            confidence = int(j[confidenceColumn])
                            ret = a
            except:
                confidence = -1

    return ret, confidence

def getHPOIDs(data : pd.DataFrame) -> list:
    ret = []

    if reduceToTestIDs:
        ret = testIDs
    else:
        if data is not None and len(data.index) > 0 and hpoidColumn in data.columns:
            ret = list(set(data[hpoidColumn].tolist()))

    return ret

def getMetrics(
        data            : pd.DataFrame = None, 
        classColumn     : str   = "", 
        answerColumn    : str   = "", 
        className       : str   = ""
) -> object:
    ret = {
        precisionLabel  : 0,
        recallLabel     : 0,
        f1ScoreLabel    : 0
    }

    if (data is not None and 
        len(classColumn) > 0 and 
        len(answerColumn) > 0 and
        len(className) > 0 and 
        classColumn in data.columns and 
        answerColumn in data.columns and 
        len(data[data[answerColumn] == className].index) > 0 and
        len(data[data[classColumn] == className].index) > 0
    ):
            ret[precisionLabel] = len(data[(data[classColumn] == className) & (
                data[answerColumn] == className)].index
                ) / (len(data[data[answerColumn] == className].index))
            
            ret[recallLabel] = len(data[(data[typeColumn] == className) & (
                data[answerColumn] == className)].index
                ) / (len(data[data[classColumn] == className].index))

            
            if ret[precisionLabel] > 0 or ret[recallLabel] > 0:
                ret[f1ScoreLabel] = (
                    (2 * ret[precisionLabel] * ret[recallLabel]) / (
                    ret[recallLabel] + 
                    ret[precisionLabel]))

    return ret
    
def cosSim(a, b) -> float:
    # a = b -> 1 (most similar)
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def eucDis(a, b) -> float:
    # the closer a and b are, the smaller the distance
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def eucSim(a, b) -> float:
    # a = b -> 1 (most similar)
    return 1 / (1 + eucDis(a, b))


def scaSim(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b)


def manDis(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.abs(a - b))


def manSim(a, b) -> float:
    # Convert distance → similarity
    return 1 / (1 + manDis(a, b))


def angSim(a, b) -> float:
    # Angular similarity = 1 - (angle / π)
    cos_sim = cosSim(a, b)
    angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
    return 1 - (angle / np.pi)


def mahDis(a, b, cov = None) -> float:
    a = np.array(a)
    b = np.array(b)
    
    diff = a - b
    
    if cov is None:
        # If no covariance matrix provided, assume identity
        cov = np.eye(len(a))
    
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))


def mahSim(a, b, cov = None) -> float:
    # Convert distance → similarity
    return 1 / (1 + mahDis(a, b, cov))

def createExampleString(examples : list, exStr : str = "Example") -> str:
    ret = "None"
    if len(examples) > 0:
        ret = ""
        for index, example in enumerate(examples):
            ret += f"{exStr} {index+1}: " + applyFormat(example) + "\n"
        ret = ret[0:len(ret) - 1]

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