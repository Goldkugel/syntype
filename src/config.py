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
configStr               = "Def" if addDefinition else "NoDef"
configStr              += "_Com" if addComment else "_NoCom"
configStr              += "_Str" if addStructure else "_NoStr"
fewShotStr              = "FS" if fewShot else "NoFS"
chainOfThoughtsStr      = "COT" if chainOfThoughts else "NoCOT"

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
# Result columns of the classification files, after the columns of the
# transformed HPO (id, attribute, value, additional)
# =============================================================================
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
outputFolderNameFormatted           = "formatted"
outputFolderNameMerged              = "merged"
outputFolderNameEvaluation          = "evaluated"


# The HPO file read by the adapter (adapter.hpo section).
inputFileTransformed        = os.path.join(
    configurationFile["adapter"]["hpo"]["input_folder"],
    configurationFile["adapter"]["hpo"]["input_files"][0]
)

# hp.owl transformed by SynTypeClassification.py, where adapter.hpo.output_folder
# and output_file say: everything the classification and the evaluation need.
# The development and evaluation set are named after it in the same folder,
# e.g. hpo.development.csv and hpo.evaluation.csv.
outputFileTransformedFull = os.path.join(
    configurationFile["adapter"]["hpo"]["output_folder"],
    configurationFile["adapter"]["hpo"]["output_file"]
)
transformedStem, transformedExtension = os.path.splitext(outputFileTransformedFull)

# The development set (syntype.development_set): the synonyms drawn at random
# and the information of their concepts, classified by runs with "test".
outputFileTransformedDevelopment = f"{transformedStem}.development{transformedExtension}"

# The evaluation set: hpo.csv without the synonyms of the development set,
# classified by runs without "test".
outputFileTransformedEvaluation  = f"{transformedStem}.evaluation{transformedExtension}"

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
        "id"        : "HP:0000011",
        "label"     : "Neurogenic bladder",
        "definition": "A type of bladder dysfunction caused by neurologic damage. Neurogenic bladder can be flaccid or spastic. Common manifestatios of neurogenic bladder are overflow incontinence, frequency, urgency, urge incontinence, and retention.",
        "comment"   : "Neurogenic bladder sphincter dysfunction (NBSD) can develop as a result of a lesion at any level in the nervous system, including the cerebral cortex, spinal cord, or peripheral nervous system. Neurologic conditions in children leading to neurogenic bladder dysfunction are predominantly congenital neural tube defects (including myelomeningocele, lipomeningocele, sacral agenesis, and occult lesions causing tethered cord). Acquired causes such as spinal cord tumors or trauma or sequelae of transverse myelitis are less frequent. Whereas from an etiologic standpoint neurogenic bladder dysfunction is a heterogeneous group, medical management will be similar irrespective of the underlying cause.",
        "parents"   : ["Functional abnormality of the bladder"],
        "children"  : [],
        "synonym"   : "Lack of bladder control due to nervous system injury",
        "type"      : laypersonSynonymType,
        "confidence": "5",
        "rationale" : "The synonym explains the clinical term 'neurogenic bladder' in everyday words ('lack of bladder control', 'nervous system injury'), as patient information does. Clinicians use the term itself."
    },
    {
        "id"        : "HP:0000056",
        "label"     : "Abnormal clitoris morphology",
        "definition": "Any structural abnormality of the clitoris.",
        "comment"   : "",
        "parents"   : ["Abnormal female external genitalia morphology"],
        "children"  : ["Bifid clitoris", "Duplicated clitoris", "Abnormal size of the clitoris", "Aplasia/Hypoplasia of the clitoris"],
        "synonym"   : "Abnormality of the clitoris",
        "type"      : expertSynonymType,
        "confidence": "4",
        "rationale" : "'Abnormality of the ...' is the descriptive phrasing of clinical documentation and phenotype databases, even though its words are plain. A patient would describe what is different, e.g. the size or shape."
    },
    {
        "id"        : "HP:0000028",
        "label"     : "Cryptorchidism",
        "definition": "Testis in inguinal canal. That is, absence of one or both testes from the scrotum owing to failure of the testis or testes to descend through the inguinal canal to the scrotum.",
        "comment"   : "The gonad is mobile and can be retracted superiorly by the cremaster muscle reflex stimulated, for instance, by cold or touch. A retracted testis is not cryptorchidism. An abdominal testis cannot be distinguished by physical examination from an (Apparently) absent testis and requires radiological (or, rarely, surgical) procedures for assessment.",
        "parents"   : ["Abnormal testis morphology"],
        "children"  : ["Bilateral cryptorchidism", "Unilateral cryptorchidism"],
        "synonym"   : "Undescended testis",
        "type"      : laypersonSynonymType,
        "confidence": "3",
        "rationale" : "The synonym describes the finding in plain words instead of the Greek-derived term 'cryptorchidism', as patients and parents say. Clinicians use it as well, so both groups share the wording."
    },
    {
        "id"        : "CHEBI:133608",
        "label"     : "1-methylhistidine zwitterion",
        "definition": "An alpha-amino acid zwitterion obtained by transfer of a proton from the carboxy to the amino group of 1-methylhistidine; major species at pH 7.3.",
        "comment"   : "",
        "parents"   : ["alpha-amino-acid zwitterion"],
        "children"  : [],
        "synonym"   : "2-azaniumyl-3-(1-methyl-1H-imidazol-4-yl)propanoate",
        "type"      : expertSynonymType,
        "confidence": "5",
        "rationale" : "The synonym is a systematic chemical name with locants and charge states, as chemistry databases and the literature use it. Patients would not use it at all."
    }
]


# The number of experiments, for the shell scripts (start.sh, finalize.sh).
if __name__ == "__main__":
    print(len(syntypeConfig.experiments))
