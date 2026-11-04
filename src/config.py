"""
The configuration of syntype: reads config/config.yaml, picks the experiment
and the models of a run from the command line, and defines the column and
class names, the file names of every step and the Few-Shot examples.
Imported by every script with `from config import *`.

    python3 ./src/config.py   prints the number of experiments
"""

import os
import sys
import yaml

from SynTypeConfig import ExperimentConfig, SyntypeConfig

# The attributes, synonym classes and source types of the transformed HPO, as
# the adapter writes them.
from adapter                  import labelClass, definitionClass, commentClass, childrenClass
from adapter                  import exactSynonymClass, relatedSynonymClass
from adapter                  import expertSynonymType, laypersonSynonymType, abbreviationSynonymType
from adapter.hpo.HPOAdapterUtils import directSynonymType

sys.dont_write_bytecode = True

# =============================================================================
# Configuration File
# =============================================================================

# The settings of syntype ("syntype" section) and of the packages it uses
# (adapter, embedder, llm, logger).
configFile = "./config/config.yaml"

with open(configFile, "r", encoding = "utf-8") as f:
    configurationFile = yaml.safe_load(f)

syntypeConfig = SyntypeConfig.model_validate(configurationFile["syntype"])

# =============================================================================
# Experiment and models of a run, given on the command line:
#   (nothing)           the preparation run (SynTypeClassification.py only);
#   <index>             the experiment syntype.experiments[<index>], from 0,
#                       with every model of llm.model_id;
#   <index> <model ID>  SynTypeClassification.py with only this model.
# =============================================================================

experimentIndex = None
if len(sys.argv) > 1:
    experimentCount = len(syntypeConfig.experiments)
    if not sys.argv[1].isdigit() or int(sys.argv[1]) >= experimentCount:
        sys.exit(f"The experiment '{sys.argv[1]}' does not exist: give its index in " \
            f"syntype.experiments, 0 to {experimentCount - 1}.")
    experimentIndex = int(sys.argv[1])

# The preparation run classifies nothing, so it has no experiment.
experiment = (ExperimentConfig() if experimentIndex is None
    else syntypeConfig.experiments[experimentIndex])

reduceToTestIDs         = experiment.test
fewShot                 = experiment.few_shot
chainOfThoughts         = experiment.chain_of_thoughts
addDefinition           = experiment.definition
addComment              = experiment.comment
addStructure            = experiment.structure

forceDrawAdditionalPlots = syntypeConfig.force_draw_additional_plots

testStr                 = "Test" if reduceToTestIDs else "NoTest"
configStr               = "Definition" if addDefinition else "NoDefinition"
configStr              += "_Comment" if addComment else "_NoComment"
configStr              += "_Structure" if addStructure else "_NoStructure"
fewShotStr              = "FewShot" if fewShot else "NoFewShot"
chainOfThoughtsStr      = "ChainOfThoughts" if chainOfThoughts else "NoChainOfThoughts"

# =============================================================================
# Model Configuration
# =============================================================================

# None in the preparation run, the one given after the experiment, or all of
# llm.model_id.
modelIDs = []
if experimentIndex is not None:
    modelIDs = sys.argv[2:3] or list(configurationFile["llm"]["model_id"])

def modelNameOf(modelID : str) -> str:
    """
    The name of a model in file names and the system column: its ID without
    the organisation, e.g. "medgemma-4b-it" for "google/medgemma-4b-it".
    """
    return modelID[modelID.index("/") + 1:] if "/" in modelID else modelID

distanceColumnPrePrefix   = "distance_"
distanceColumnPrefix      = f"{distanceColumnPrePrefix}" + "{}_{}"

distanceEvaluationLowerBound = syntypeConfig.distance_evaluation_lower_bound
distanceEvaluationUpperBound  = syntypeConfig.distance_evaluation_upper_bound
distanceEvaluationParts      = int((distanceEvaluationUpperBound - distanceEvaluationLowerBound) * 1000 + 1)

# Mean and standard deviation of the cosine distance with which the distances
# are normalised (None: computed from the data).
cosineDistanceMean           = syntypeConfig.cosine_distance_mean
cosineDistanceStd            = syntypeConfig.cosine_distance_std

# =============================================================================
# For Simplification
# =============================================================================

quotationCharacter = "\""

# =============================================================================
# Column names of the synonym tables and classification files
# =============================================================================
hpoidColumn                 = "hpoID"
classColumn                 = "class"
typeColumn                  = "type"
contentColumn               = "content"
systemColumn                = "system"
answerColumn                = "answer"
confidenceColumn            = "confidence"
# The short explanation of a classification in the LLM's answer.
rationaleColumn             = "rationale"

# =============================================================================
# The answer key, and the synonym classes and types syntype classifies
# =============================================================================

# The key of the classification in the LLM's JSON answer.
classificationKey               = "classification"

# The synonym classes syntype classifies; broad and narrow synonyms name
# another phenotype than their concept.
synonymClasses = [exactSynonymClass, relatedSynonymClass]

undefinedSynonymType            = "undefined"

# The types the LLM classifies synonyms into (see prompts/).
synonymTypes = [expertSynonymType, laypersonSynonymType]

goldStandardSystem              = "gold"

precisionLabel  = "precision"
recallLabel     = "recall"
f1ScoreLabel    = "f1"

# =============================================================================
# Folder structure
# =============================================================================

csvFileFormat = "csv"
plotFileFormat = "png"

dataDir = syntypeConfig.data_folder

outputFolderName                    = "output"

# The folders of the steps: transformation (and the development and
# evaluation set), formatting, merging and evaluation; the raw classification
# is in outputFolderNameClass, the distances in outputFolderNameEmbedded.
outputFolderNameTransformed         = "transformed"
outputFolderNameFormatted           = "formatted"
outputFolderNameMerged              = "merged"
outputFolderNameEvaluation          = "evaluated"


# The HPO file read by the adapter (adapter.hpo section).
inputFileTransformed        = os.path.join(
    configurationFile["adapter"]["hpo"]["input_folder"],
    configurationFile["adapter"]["hpo"]["input_files"][0]
)

# hp.owl transformed by SynTypeClassification.py: everything the
# classification and the evaluation need.
outputFileNameTransformedFull       = f"hpo.{csvFileFormat}"

outputFileTransformedFull                     = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameTransformed,
    outputFileNameTransformedFull
)

# The development set (syntype.development_set): the synonyms drawn at random
# and the information of their concepts, classified by runs with "test".
outputFileNameTransformedDevelopment = f"hpo.development.{csvFileFormat}"

outputFileTransformedDevelopment     = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameTransformed,
    outputFileNameTransformedDevelopment
)

# The evaluation set: hpo.csv without the synonyms of the development set,
# classified by runs without "test".
outputFileNameTransformedEvaluation  = f"hpo.evaluation.{csvFileFormat}"

outputFileTransformedEvaluation      = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameTransformed,
    outputFileNameTransformedEvaluation
)

inputFileTask = outputFileTransformedDevelopment if reduceToTestIDs else outputFileTransformedEvaluation

# Distances of the synonyms to their label (the preparation run): of the
# development set, for the evaluation of test runs.
outputFolderNameEmbedded             = "embedded"
outputFileEmbeddedDevelopment        = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEmbedded,
    f"embed.development.{csvFileFormat}"
)

# Of the evaluation set, for the evaluation of the other runs.
outputFileEmbeddedEvaluation         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEmbedded,
    f"embed.{csvFileFormat}"
)

inputFileDistances = outputFileEmbeddedDevelopment if reduceToTestIDs else outputFileEmbeddedEvaluation

# =============================================================================
# Files for Synonym Classification
# =============================================================================

outputFolderNameClass    = "type"
filesPrefixClass         = f"{outputFolderNameClass}_{testStr}_{configStr}"
filesPrefixApproachClass = f"{filesPrefixClass}_{chainOfThoughtsStr}_{fewShotStr}"



inputFileClass                         = inputFileTask

outputFileNameClassFormattedPrefix     = f"{filesPrefixApproachClass}_formatted"

def outputFileClass(modelName : str) -> str:
    """
    The raw classification of a model: prompts, answers and the parsed
    classification.
    """
    return os.path.join(dataDir, outputFolderName, outputFolderNameClass,
        f"{filesPrefixApproachClass}_raw_{modelName}.{csvFileFormat}")

def outputFileClassFormatted(modelName : str) -> str:
    """
    The formatted classification of a model, merged by SynTypeMerge.py.
    """
    return os.path.join(dataDir, outputFolderName, outputFolderNameFormatted,
        f"{outputFileNameClassFormattedPrefix}_{modelName}.{csvFileFormat}")

# Read by SynTypeMerge.py.
inputFolderClassMerged                 = os.path.join(dataDir, outputFolderName, outputFolderNameFormatted)

outputFileNameClassMerged              = f"{filesPrefixApproachClass}_merged_classes.{csvFileFormat}"
outputFileClassMerged                  = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameMerged,
    outputFileNameClassMerged
)

inputFileClassEvaluation               = outputFileClassMerged

outputFileNameClassGoldCounts   = f"{filesPrefixApproachClass}_gold_counts.{plotFileFormat}"
outputFileClassGoldCounts       = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassGoldCounts
)

outputFileNameClassRecallPrecisionF1   = f"{filesPrefixApproachClass}_base_evaluation.{plotFileFormat}"
outputFileClassRecallPrecisionF1 = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassRecallPrecisionF1
)

outputFileNameClassAnswerCounts   = f"{filesPrefixApproachClass}_answer_counts.{plotFileFormat}"
outputFileClassAnswerCounts = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassAnswerCounts
)

outputFileNameClassEvaluationExactHPO     = f"{filesPrefixApproachClass}_exact_HPO_evaluation.{plotFileFormat}"
outputFileClassEvaluationExactHPO         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExactHPO
)

outputFileNameClassEvaluationExactUBERON     = f"{filesPrefixApproachClass}_exact_UBERON_evaluation.{plotFileFormat}"
outputFileClassEvaluationExactUBERON         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExactUBERON
)
outputFileNameClassEvaluationExactGO     = f"{filesPrefixApproachClass}_exact_GO_evaluation.{plotFileFormat}"
outputFileClassEvaluationExactGO         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExactGO
)
outputFileNameClassEvaluationExactCHEBI     = f"{filesPrefixApproachClass}_exact_CHEBI_evaluation.{plotFileFormat}"
outputFileClassEvaluationExactCHEBI         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExactCHEBI
)

outputFolderNameClassEmbedding = outputFolderNameEvaluation

embeddingPrefix = f"{outputFolderNameClass}_{testStr}"

outputFileNameClassEmbeddingEvaluation    = f"{embeddingPrefix}_embedding_evaluation_" + "{}" + f".{plotFileFormat}"
outputFileNameClassDistanceEvaluation   = f"{embeddingPrefix}_distance_evaluation_" + "{}" + f".{plotFileFormat}"

# The summary table of the models and baselines, and the confusion matrices,
# ROC curves and confidence calibration of the models.
def outputFileClassEvaluationOf(name : str, fileFormat : str = plotFileFormat) -> str:
    """The path of an evaluation output of the experiment, e.g. ..._summary.csv."""
    return os.path.join(dataDir, outputFolderName, outputFolderNameEvaluation,
        f"{filesPrefixApproachClass}_{name}.{fileFormat}")

outputFileClassSummary     = outputFileClassEvaluationOf("summary", csvFileFormat)
outputFileClassConfusion   = outputFileClassEvaluationOf("confusion_matrices")
outputFileClassROC         = outputFileClassEvaluationOf("roc")
outputFileClassCalibration = outputFileClassEvaluationOf("calibration")


# The Few-Shot examples with their answer (classification, confidence from 1
# to 5, rationale). Their synonyms are left out of the evaluation also in runs
# without examples, so that all runs are scored on the same synonyms.
fewShotExamples = [
    {
        hpoidColumn : "HP:0008365",
        "label"     : "Abnormal talus morphology",
        "definition": "An abnormality of the talus.",
        "comment"   : "",
        "parents"   : ["Abnormality of the tarsal bones"],
        "children"  : ["Tarsal osteovalgus", "Talar aplasia", "Shortening of the talar neck", "Rocker bottom foot", "Osteolysis of talus", "Osteochondral lesion of talus", "Os trigonum", "Fractured talus", "Delayed talus ossification"],
        "synonym"   : "Abnormal large bone of ankle",
        "type"      : laypersonSynonymType,
        "confidence": "5",
        "rationale" : "The synonym replaces the anatomical term 'talus' with the plain description 'large bone of ankle', as patient-facing materials do. Clinicians and phenotype databases name the talus directly."
    },
    {
        hpoidColumn : "HP:0000132",
        "label"     : "Menorrhagia",
        "definition": "Prolonged and excessive menses at regular intervals in excess of 80 mL or lasting longer than 7 days.",
        "comment"   : "",
        "parents"   : ["Abnormal bleeding", "Abnormality of the menstrual cycle"],
        "children"  : [],
        "synonym"   : "Hypermenorrhea",
        "type"      : expertSynonymType,
        "confidence": "5",
        "rationale" : "'Hypermenorrhea' is a Greek-derived medical term used in clinical documentation and gynaecological literature. Patients would speak of heavy or long periods instead."
    },
    {
        hpoidColumn : "HP:0000218",
        "label"     : "High palate",
        "definition": "Height of the palate more than 2 SD above the mean (objective) or palatal height at the level of the first permanent molar more than twice the height of the teeth (subjective).",
        "comment"   : "The measuring device for this assessment is described in (Hall JG, Froster-Iskenius UG, Allanson JE, Gripp K, Slavotinek A. 2006. Handbook of Normal Physical Measurements. 2nd edition. Oxford Medical, publishers). A high palate is often associated with a narrow palate. However, a narrow palate can easily give a false appearance of a high palate. Height and width of the palate should be assessed and coded separately. We do not recommend the subjective determination because this term can be overused and applied inaccurately.",
        "parents"   : ["Abnormal palate morphology"],
        "children"  : ["High, narrow palate"],
        "synonym"   : "Increased palatal height",
        "type"      : laypersonSynonymType,
        "confidence": "4",
        "rationale" : "The synonym paraphrases the finding as an increased height of the palate instead of using the established clinical term 'high palate', as explanations for patients do. The technical word 'palatal' makes the category likely rather than certain."
    }
]

# The number of experiments, for the shell scripts (start.sh, finalize.sh).
if __name__ == "__main__":
    print(len(syntypeConfig.experiments))
