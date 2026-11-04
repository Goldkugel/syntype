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

# Prevent Python from generating .pyc files (compiled bytecode files)
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
# Experiment and models of a run, given on the command line (see start.sh,
# classify.sh and finalize.sh):
#   (nothing)           SynTypeClassification.py: the preparation run
#                       (transforms the HPO, draws the development set
#                       and embeds both);
#   <index>             the experiment syntype.experiments[<index>], counted
#                       from 0: SynTypeClassification.py classifies the
#                       synonyms with every model of llm.model_id, one after
#                       the other; SynTypeMerge.py and SynTypeEval.py merge
#                       and evaluate the classifications of all models;
#   <index> <model ID>  SynTypeClassification.py: the same with only this
#                       model.
# `python3 ./src/config.py` prints the number of experiments.
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

# The LLMs of SynTypeClassification.py, classifying one after the other: none
# in the preparation run, the one given after the experiment, or all of
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
# For Data Curation
# =============================================================================
sourceLanguageShort = "en"
sourceLanguage      = "English"

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
# Data Classes of Concepts in HPO that are being processed
# =============================================================================

labelClass                      = "label"
definitionClass                 = "definition"
commentClass                    = "comment"
referenceClass                  = "reference"

synonymClass                    = "classification"
exactSynonymClass               = "exact"
relatedSynonymClass             = "related"
broadSynonymClass               = "broad"
narrowSynonymClass              = "narrow"

# The synonym classes syntype classifies; broad and narrow synonyms name
# another phenotype than their concept.
synonymClasses = [exactSynonymClass, relatedSynonymClass]

expertSynonymType               = "expert"
laypersonSynonymType            = "layperson"
abbreviationSynonymType         = "abbreviation"
obsoleteSynonymType             = "obsolete"
pluralFormSynonymType           = "plural"
ukSpellingSynonymType           = "uk"
allelicRequirementSynonymType   = "allelic"
# In OWL Class Section, rather than in Axiom Section.
directSynonymType               = "direct"

undefinedSynonymType            = "undefined"

# The types the LLM classifies synonyms into (see prompts/).
synonymTypes = [expertSynonymType, laypersonSynonymType]

childrenClass                   = "child"

enrichedSourceExactSynonymClass = "generatedSynonym"
enrichedSourceDefinitionClass   = "generatedDefinition"

goldStandardSystem              = "gold"

precisionLabel  = "precision"
recallLabel     = "recall"
accuracyLabel   = "accuracy"
f1ScoreLabel    = "f1"

# =============================================================================
# Folder structure
# =============================================================================

csvFileFormat = "csv"
pickleFileFormat = "pkl"
logFileFormat = "log"
plotFileFormat = "png"

# Basic Data Directory.
dataDir = syntypeConfig.data_folder

# Basic Data Subdirectories.
outputFolderName                    = "output"
logFolderName                       = "logs"

# Basic Data Processing Directories.
# First step (and the embeddings for the evaluation).
outputFolderNameTransformed         = "transformed"
# The second Step is the actual job e.g. generation or classification.
# Third step.
outputFolderNameFormatted           = "formated"
# Fourth step.
outputFolderNameMerged              = "merged"
# Fifth step.
outputFolderNameEvaluation          = "evaluated"


# The HPO file read by the adapter (adapter.hpo section).
inputFileTransformed        = os.path.join(
    configurationFile["adapter"]["hpo"]["input_folder"],
    configurationFile["adapter"]["hpo"]["input_files"][0]
)

# Contains all Information of the hp.owl needed to Generate and Classify 
# Synonyms and perform the Evaluation.
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

# Cosine distances of the synonyms to their label (the preparation run): of the
# development set, for the evaluation of test runs; their mean and standard
# deviation are logged, to be set as syntype.cosine_distance_mean and _std.
outputFolderNameEmbedded             = "embeded"
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

# The formatted classifications of all models of this experiment are read
# from here by SynTypeMerge.py.
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


# =============================================================================

# The examples of the Few-Shot approach, with the answer of the prompts
# (classification, confidence from 1 to 5, rationale). Their synonyms are
# left out of the evaluation (SynTypeEval.py; SynTypeClassification.py writes
# them into the prompts), also in runs without examples, so that all runs are
# scored on the same synonyms.
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
