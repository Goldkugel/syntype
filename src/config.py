import os
import sys
import torch

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

reduceToTestIDs         = False
if "test" in sys.argv:
    reduceToTestIDs = True

fewShot                 = False
chainOfThoughts         = False

if "few-shot" in sys.argv:
    fewShot = True

if "chain-of-thoughts" in sys.argv:
    chainOfThoughts = True

forceDrawAdditionalPlots = True

chunkSizeAnswerGeneration = 500

addDefinition   = False
if "definition" in sys.argv:
    addDefinition = True

addComment      = False
if "comment" in sys.argv:
    addComment = True

addParents      = False
if "parents" in sys.argv:
    addParents = True

addChildren     = False
if "children" in sys.argv:
    addChildren = True

testStr                 = "Test" if reduceToTestIDs else "NoTest"
configStr               = "Definition" if addDefinition else "NoDefinition"
configStr              += "_Comment" if addComment else "_NoComment"
configStr              += "_Parents" if addParents else "_NoParents"
configStr              += "_Children" if addChildren else "_NoChildren"
fewShotStr              = "FewShot" if fewShot else "NoFewShot"
chainOfThoughtsStr      = "ChainOfThoughts" if chainOfThoughts else "NoChainOfThoughts"

# =============================================================================
# Model Configuration
# =============================================================================

modelID = ""
modelName = ""
if len(sys.argv) > 1 and len(sys.argv[1]) > 0 and sys.argv[1][0] != "-":
    modelID = sys.argv[1]
    modelName = modelID[modelID.index("/") + 1:]

# Possible Similarity Metrics
cosineSimilarity        = "cosine"
#euclideanSimilarity     = "euclidean"
#scalarSimilarity        = "scalar"
#manhattanSimilarity     = "manhattan"
#angularSimilarity       = "angular"
#mahalanobisSimilarity   = "mahalanobis"

# Similarity Metrics used. Scalar Similarity is not very useful. Additionally,
# Mahalanobis Similarity is not necessary, if afterwards all metrics are
# normalized. 
similarityMetrics = [
    cosineSimilarity,
    #euclideanSimilarity,
    #manhattanSimilarity,
    #angularSimilarity
]

# Model Names used to embed the synonyms and labels and to generate 
# similarities.
bioClinicalBERT = "BioClinicalBERT"
bioBERT         = "BioBERT"
clinicalBERT    = "ClinicalBERT"
sapBERT         = "SapBERT"
sciBERT         = "SciBERT"
umlsBERT        = "UMLSBERT"
sapUMLSBERT     = "SapUMLSBERT"
medCPT          = "medCPT"
pubmedBERT      = "PubMedBERT"
bioLinkBERT     = "BioLinkBERT"

# The model IDs for the embeddings.
embeddingModels = {
    #bioClinicalBERT : "emilyalsentzer/Bio_ClinicalBERT",
    sapBERT         : "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"#,
    #clinicalBERT    : "medicalai/ClinicalBERT",
    #bioBERT         : "dmis-lab/biobert-v1.1",
    #umlsBERT        : "GanjinZero/UMLSBert_ENG",
    #sciBERT         : "allenai/scibert_scivocab_cased",
    #sapUMLSBERT     : "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
    #medCPT          : "ncbi/MedCPT-Query-Encoder",
    #pubmedBERT      : "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    #bioLinkBERT     : "michiyasunaga/BioLinkBERT-large"
}

similarityColumnPrePrefix   = "similarity_"
similarityColumnPrefix      = f"{similarityColumnPrePrefix}" + "{}_{}"

embeddingThresholdsLayperson = {
    similarityColumnPrefix.format(sapBERT,         cosineSimilarity) : -0.1
}

embeddingThresholdsExpert = {
    similarityColumnPrefix.format(sapBERT,         cosineSimilarity) : 1.399 
}

similarityEvaluationLowerBound = -4
similarityEvaluationUperBound  = 4
similarityEvaluationParts      = (similarityEvaluationUperBound - similarityEvaluationLowerBound) * 1000 + 1

gpus = int(torch.cuda.device_count())
gpu_id = ""
if gpus > 0:
    gpu_id = ','.join(map(str, [i for i in range(0, gpus)]))

# Float that controls the cumulative probability of the top tokens to consider.
# Must be in (0, 1]. Set to 1 to consider all tokens.
top_p=0.95

max_model_len = 4 * 2048
max_num_batched_tokens = 2 * max_model_len

# Float that controls the randomness of the sampling. Lower values make the 
# model more deterministic, while higher values make the model more random. 
# Zero means greedy sampling.
temperature = 0.01

# Maximum number of tokens to generate per output sequence.
max_tokens = 2048

# Random seed to use for the generation
seed = 2898231092

# =============================================================================
# For Simplification
# =============================================================================

quotationCharacter = "\""

systemRole = "system"
userRole = "user"
modelRole = "assistant"

# For Gemma
startTurnID = "start_of_turn"
endTurnID = "end_of_turn"

# For Llama
startHeaderID = "start_header_id"
endHeaderID = "end_header_id"
endOfTextID = "eot_id"
beginOfTextID = "begin_of_text"
endOfTextID2 = "end_of_text"

startTag = "<"
endTag = ">"
bar = "|"

unusedTokens = "<unused95>"

# For Gemma
startTurn = f"{startTag}{startTurnID}{endTag}"
endTurn = f"{startTag}{endTurnID}{endTag}"

# For Llama
startHeader = f"{startTag}{bar}{startHeaderID}{bar}{endTag}"
endHeader = f"{startTag}{bar}{endHeaderID}{bar}{endTag}"
endOfText = f"{startTag}{bar}{endOfTextID}{bar}{endTag}"
beginOfText = f"{startTag}{bar}{beginOfTextID}{bar}{endTag}"
endOfText2 = f"{startTag}{bar}{endOfTextID2}{bar}{endTag}"

messageRoleElement = "role"
messageTextElement = "message"

headerChar = "="
headerLen = 60
headerSeparator = headerChar * headerLen

progressBarColor = "cyan"
progressBarTextLength = 40

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
roundColumn                 = "round"
answerColumn                = "answer"
confidenceColumn            = "confidence"

embeddingColumn             = "embedding"

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

synonymClasses = [exactSynonymClass, relatedSynonymClass]#, broadSynonymClass, narrowSynonymClass]

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

synonymTypes = [expertSynonymType, laypersonSynonymType]

childrenClass                   = "child"

enrichedSourceExactSynonymClass = "generatedSynonym"
enrichedSourceDefinitionClass   = "generatedDefinition"

goldStandardSystem              = "gold"

owlSourceExactSynonym                   = "hasExactSynonym"
owlSourceRelatedSynonym                 = "hasRelatedSynonym"
owlSourceBoradSynonym                   = "hasBroadSynonym"
owlSourceNarrowSynonym                  = "hasNarrowSynonym"

owlSourceSynonymTypeLayperson           = "layperson"
owlSourceSynonymTypeAbbreviation        = "abbreviation"
owlSourceSynonymTypeObsolete            = "obsolete_synonym"
owlSourceSynonymTypePlural              = "plural_form"
owlSourceSynonymTypeUKSpelling          = "uk_spelling"
owlSourceSynonymTypeAllelic             = "allelic_requirement"

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
dataDir = "../data"

# Basic Data Subdirectories.
inputFolderName                     = "input"
outputFolderName                    = "output"
logFolderName                       = "logs"

# Basic Data Processing Directories.
# First step.
outputFolderNameTransformed         = "transform"
# The second Step is the actual job e.g. generation or classification.
# Third step.
outputFolderNameFormatted           = "format"
# Fourth step.
outputFolderNameMerged              = "merge"
# The gold standards are saved here, might not be necessary.
outputFolderNameGold                = "gold"
# Fifth step.
outputFolderNameEvaluation          = "evaluate"

logFileName                         = f"syntype_{testStr}"
if modelName != "":
    logFileName                     += f"_{configStr}_{chainOfThoughtsStr}_{fewShotStr}_{modelName}"
logFileName                         += f".{logFileFormat}"

logFilePromptsName                  = f"prompts_{testStr}_{configStr}_{chainOfThoughtsStr}_{fewShotStr}_{modelName}.{logFileFormat}"

logFile                     = os.path.join(
    dataDir,
    logFolderName,
    logFileName
)

logFilePrompts              = os.path.join(
    dataDir,
    logFolderName,
    logFilePromptsName
)

# The Input Folders of each Step
inputFolderNameTransformed          = inputFolderName

inputFileNameTransformed            = "hp.owl"

inputFileTransformed        = os.path.join(
    dataDir,
    inputFolderName,
    inputFileNameTransformed
)

# Contains all Information of the hp.owl needed to Generate and Classify 
# Synonyms and perform the Evaluation.
outputFileNameTransformedFull       = f"transform.{csvFileFormat}"

outputFileTransformedFull                     = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameTransformed,
    outputFileNameTransformedFull
)

# Reduction to Test HPO IDs listed at the bottom for engineering purposes.
outputFileNameTransformed           = f"transform.reduced.{csvFileFormat}"

outputFileTransformed                        = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameTransformed,
    outputFileNameTransformed
)

inputFileTask = outputFileTransformed if reduceToTestIDs else outputFileTransformedFull

# =============================================================================
# Files for Synonym Classification
# =============================================================================

outputFolderNameClass    = "type"
filesPrefixClass         = f"{outputFolderNameClass}_{testStr}_{configStr}"
filesPrefixApproachClass = f"{filesPrefixClass}_{chainOfThoughtsStr}_{fewShotStr}"



inputFileClass                         = inputFileTask

outputFileNameClass                    = f"{filesPrefixApproachClass}_raw_{modelName}.{csvFileFormat}"
outputFileClass                        = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameClass,
    outputFileNameClass
)

# Contains the formatted classified Synonyms of the Model. 
inputFileClassFormatted                = outputFileClass
outputFileNameClassFormattedPrefix     = f"{filesPrefixApproachClass}_formatted"
outputFileNameClassFormatted           = f"{outputFileNameClassFormattedPrefix}_{modelName}.{csvFileFormat}"
outputFileClassFormatted               = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameFormatted,
    outputFileNameClassFormatted
)

inputFileNameClassMerged               = [
    file
    for file in os.listdir(os.path.join(dataDir, outputFolderName, outputFolderNameFormatted))
    if file.startswith(outputFileNameClassFormattedPrefix) and file.endswith(csvFileFormat)
]
inputFileClassMerged                   = [
    os.path.join(dataDir, outputFolderName, outputFolderNameFormatted, filename) for filename in inputFileNameClassMerged
]

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

outputFileNameClassEvaluationExpert     = f"{filesPrefixApproachClass}_expert_evaluation.{plotFileFormat}"
outputFileClassEvaluationExpert         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExpert
)

outputFileNameClassEvaluationExpertHPO     = f"{filesPrefixApproachClass}_expert_HPO_evaluation.{plotFileFormat}"
outputFileClassEvaluationExpertHPO         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExpertHPO
)

outputFileNameClassEvaluationExpertUBERON     = f"{filesPrefixApproachClass}_expert_UBERON_evaluation.{plotFileFormat}"
outputFileClassEvaluationExpertUBERON         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExpertUBERON
)
outputFileNameClassEvaluationExpertGO     = f"{filesPrefixApproachClass}_expert_GO_evaluation.{plotFileFormat}"
outputFileClassEvaluationExpertGO         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExpertGO
)
outputFileNameClassEvaluationExpertCHEBI     = f"{filesPrefixApproachClass}_expert_CHEBI_evaluation.{plotFileFormat}"
outputFileClassEvaluationExpertCHEBI         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassEvaluationExpertCHEBI
)

outputFileNameClassAccuracyMacroMicro  = f"{filesPrefixApproachClass}_accuracy_threshold.{plotFileFormat}"
outputFileClassAccuracyMacroMicro      = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassAccuracyMacroMicro
)

outputFileNameClassClassAccuracy       = f"{filesPrefixApproachClass}_class_accuracy.{plotFileFormat}"
outputFileClassClassAccuracy           = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassClassAccuracy
)

outputFileNameCombinedEvaluationAbsolute       = f"{filesPrefixApproachClass}_combined_evaluation_absolute.{plotFileFormat}"
outputFileCombinedEvaluationAbsolute           = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameCombinedEvaluationAbsolute
)

outputFileNameCombinedEvaluationAbsolute2      = f"{filesPrefixApproachClass}_combined_evaluation_absolute2.{plotFileFormat}"
outputFileCombinedEvaluationAbsolute2          = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameCombinedEvaluationAbsolute2
)

outputFileNameCombinedEvaluationRelaxed       = f"{filesPrefixApproachClass}_combined_evaluation_relaxed.{plotFileFormat}"
outputFileCombinedEvaluationRelaxed           = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameCombinedEvaluationRelaxed
)

outputFolderNameClassEmbedding = outputFolderNameEvaluation

embeddingPrefix = f"{outputFolderNameClass}_{testStr}"

outputFileNameClassEmbeddingEvaluation    = f"{embeddingPrefix}_embedding_evaluation_" + "{}" + f".{plotFileFormat}"
outputFileNameClassSimilarityEvaluation   = f"{embeddingPrefix}_similarity_evaluation_" + "{}" + f".{plotFileFormat}"

outputFileNameClassEmbeddingSSMD          = f"{embeddingPrefix}_embedding_ssmd.{plotFileFormat}"
outputFileClassEmbeddingSSMD              = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingSSMD
)

outputFileNameClassEmbeddingOutcomeCounts = f"{embeddingPrefix}_embedding_outcome_counts.{plotFileFormat}"
outputFileClassEmbeddingOutcomeCounts     = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingOutcomeCounts
)

outputFileNameClassEmbeddingDifference    = f"{embeddingPrefix}_embedding_diff_votes.{plotFileFormat}"
outputFileClassEmbeddingDifference        = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingDifference
)

outputFileNameClassEmbeddingExpertRelaxed    = f"{embeddingPrefix}_embedding_expert_votes_relaxed.{plotFileFormat}"
outputFileClassEmbeddingExpertRelaxed        = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingExpertRelaxed
)

outputFileNameClassEmbeddingLaypersonAbsolute    = f"{embeddingPrefix}_embedding_layperson_votes_absolute.{plotFileFormat}"
outputFileClassEmbeddingLaypersonAbsolute        = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingLaypersonAbsolute
)

outputFileNameClassEmbeddingExpertAbsolute    = f"{embeddingPrefix}_embedding_expert_votes_absolute.{plotFileFormat}"
outputFileClassEmbeddingExpertAbsolute        = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingExpertAbsolute
)

outputFileNameClassEmbeddingLaypersonRelaxed    = f"{embeddingPrefix}_embedding_layperson_votes_relaxed.{plotFileFormat}"
outputFileClassEmbeddingLaypersonRelaxed        = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingLaypersonRelaxed
)

outputFileNameClassEmbeddingPerformance   = f"{embeddingPrefix}_embedding_performance.{plotFileFormat}"
outputFileClassEmbeddingPerformance       = os.path.join(
    dataDir,        
    outputFolderName,
    outputFolderNameClassEmbedding,
    outputFileNameClassEmbeddingPerformance
)


# =============================================================================
testIDs = list(
    set([
    "HP:0005372",
    "HP:0001618",
    "HP:0002825",
    "HP:0100402",
    "HP:0006638",
    "HP:0009413",
    "HP:0000859",
    "HP:0002669",
    "HP:0100929",
    "HP:0011070",
    "HP:0002293",
    "HP:0003302",
    "HP:0006682",
    "HP:0010463",
    "HP:0001195",
    "HP:0001357",
    "HP:0001810",
    "HP:0002028",
    "HP:0006289",
    "HP:0005280",
    "HP:0001566",
    "HP:0001852",
    "HP:0010981",
    "HP:0007410",
    "HP:0100213",
    "HP:0100485",
    "HP:0010394",
    "HP:0000073",
    "HP:0000418",
    "HP:0010182",
    "HP:0000636",
    "HP:0002031",
    "HP:0004370",
    "HP:0009953",
    "HP:0005274",
    "HP:0001531",
    "HP:0002647",
    "HP:0100189",
    "HP:0008056",
    "HP:0002810",
    "HP:0005216",
    "HP:0009374",
    "HP:0010058",
    "HP:0010001",
    "HP:0000519",
    "HP:0100163",
    "HP:0006315",
    "HP:0002594",
    "HP:0100049",
    "HP:0000159",
    "HP:0009778",
    "HP:0006313",
    "HP:0009299",
    "HP:0009938",
    "HP:0010076",
    "HP:0000298",
    "HP:0005101",
    "HP:0100550",
    "HP:0000998",
    "HP:0007083",
    "HP:0040189",
    "HP:0000973",
    "HP:0000751",
    "HP:0100914",
    "HP:0000823",
    "HP:0004438",
    "HP:0006292",
    "HP:0010003",
    "HP:0010414",
    "HP:0009439",
    "HP:0004681",
    "HP:0011957",
    "HP:0009155",
    "HP:0007732",
    "HP:0009383",
    "HP:0009228",
    "HP:0000540",
    "HP:0100540",
    "HP:0002034",
    "HP:0012789",
    "HP:0000062",
    "HP:0009823",
    "HP:0010271",
    "HP:0010148",
    "HP:0011062",
    "HP:0000464",
    "HP:0009293",
    "HP:0005613",
    "HP:0000256",
    "HP:0005590",
    "HP:0003121",
    "HP:0004488",
    "HP:0010126",
    "HP:0004219",
    "HP:0008065",
    "HP:0009161",
    "HP:0002986",
    "HP:0012583",
    "HP:0200021",
    "HP:0012757",
    "HP:0009215",
    "HP:0009167",
    "HP:0100048",
    "HP:0006339",
    "HP:0000750",
    "HP:0009554",
    "HP:0200122",
    "HP:0001476",
    "HP:0000643",
    "HP:0009237",
    "HP:0009192",
    "HP:0012810",
    "HP:0500157",
    "HP:0009150",
    "HP:0002218",
    "HP:0001382",
    "HP:0100336",
    "HP:0001347",
    "HP:0005537",
    "HP:0009580",
    "HP:0000277",
    "HP:0011855",
    "HP:0012730",
    "HP:0200161",
    "HP:0000929",
    "HP:0000602",
    "HP:0004313",
    "HP:0010111",
    "HP:0010388",
    "HP:0500142",
    "HP:0100074",
    "HP:0010650",
    "HP:0012434",
    "HP:0010885",
    "HP:0100903",
    "HP:0011048",
    "HP:0001335",
    "HP:0006332",
    "HP:0007340",
    "HP:0005272",
    "HP:0004334",
    "HP:0003861",
    "HP:0000413",
    "HP:0009931",
    "HP:0001631",
    "HP:0100233",
    "HP:0010721",
    "HP:0001156",
    "HP:0011228",
    "HP:0009240",
    "HP:0200007",
    "HP:0000309",
    "HP:0001344",
    "HP:0010195",
    "HP:0010392",
    "HP:0002088",
    "HP:0001807",
    "HP:0009334",
    "HP:0009217",
    "HP:0009458",
    "HP:0001436",
    "HP:0011478",
    "HP:0010059",
    "HP:0012443",
    "HP:0012444",
    "HP:0003895",
    "HP:0010587",
    "HP:0100484",
    "HP:0012733",
    "HP:0003860",
    "HP:0000474",
    "HP:0010286",
    "HP:0003025",
    "HP:0012746",
    "HP:0007500",
    "HP:0011341",
    "HP:0100928",
    "HP:0100448",
    "HP:0003194",
    "HP:0010144",
    "HP:0010011",
    "HP:0011081",
    "HP:0001769",
    "HP:0012809",
    "HP:0001572",
    "HP:0000123",
    "HP:0000079",
    "HP:0000327",
    "HP:0008783",
    "HP:0000700",
    "HP:0002236",
    "HP:0005991",
    "HP:0009602",
    "HP:0100154",
    "HP:0006480",
    "HP:0001838",
    "HP:0002286",
    "HP:0500163",
    "HP:0012047",
    "HP:0009558",
    "HP:0100441",
    "HP:0002465",
    "HP:0045006",
    "HP:0000766",
    "HP:0100739",
    "HP:0007626",
    "HP:0011083",
    "HP:0031467",
    "HP:0004428",
    "HP:0030430",
    "HP:0004218",
    "HP:0030319",
    "HP:0100171",
    "HP:0100060",
    "HP:0009376",
    "HP:0100615",
    "HP:0000653",
    "HP:0012479",
    "HP:0010896",
    "HP:0001804",
    "HP:0000431",
    "HP:0011340",
    "HP:0100481",
    "HP:0009994",
    "HP:0004234",
    "HP:0030213",
    "HP:0001839",
    "HP:0009582",
    "HP:0009393",
    "HP:0009244",
    "HP:0100438",
    "HP:0003108",
    "HP:0009058",
    "HP:0001809",
    "HP:0010341",
    "HP:0000252",
    "HP:0010608",
    "HP:0000248",
    "HP:0001249",
    "HP:0004791",
    "HP:0012019",
    "HP:0011863",
    "HP:0040079",
    "HP:0002033",
    "HP:0002750",
    "HP:0010043",
    "HP:0003071",
    "HP:0003717",
    "HP:0010570",
    "HP:0040024",
    "HP:0000708",
    "HP:0000194",
    "HP:0006347",
    "HP:0033055",
    "HP:0007311",
    "HP:0006311",
    "HP:0100937",
    "HP:0001595",
    "HP:0004416",
    "HP:0003510",
    "HP:0030143",
    "HP:0002982",
    "HP:0010243",
    "HP:0001114",
    "HP:0012653",
    "HP:0000389",
    "HP:0000526",
    "HP:0003724",
    "HP:0010535",
    "HP:0009564",
    "HP:0009378",
    "HP:0006094",
    "HP:0003848",
    "HP:0005465",
    "HP:0040181",
    "HP:0030867",
    "HP:0003513",
    "HP:0009316",
    "HP:0005257",
    "HP:0010016",
    "HP:0000601",
    "HP:0009958",
    "HP:0012461",
    "HP:0100658",
    "HP:0009321",
    "HP:0025223",
    "HP:0001272",
    "HP:0012569",
    "HP:0000845",
    "HP:0009674",
    "HP:0033746",
    "HP:0002808",
    "HP:0000974",
    "HP:0011304",
    "HP:0009379",
    "HP:0000421",
    "HP:0000656",
    "HP:0005322",
    "HP:0004406",
    "HP:0006336",
    "HP:0010571",
    "HP:0001167",
    "HP:0011831",
    "HP:0009159",
    "HP:0011082",
    "HP:0002907",
    "HP:0010385",
    "HP:0011072",
    "HP:0200153",
    "HP:0004322",
    "HP:0100362",
    "HP:0100409",
    "HP:0000141",
    "HP:0010741",
    "HP:0002947",
    "HP:0012808",
    "HP:0000178",
    "HP:0003233",
    "HP:0011338",
    "HP:0002054",
    "HP:0009989",
    "HP:0006239",
    "HP:0000260",
    "HP:0100487",
    "HP:0009387",
    "HP:0004112",
    "HP:0000674",
    "HP:0000978",
    "HP:0003841",
    "HP:0003066",
    "HP:0002905",
    "HP:0410012",
    "HP:0004207",
    "HP:0004415",
    "HP:0033841",
    "HP:0100639",
    "HP:0001510",
    "HP:0009016",
    "HP:0002166",
    "HP:0005997",
    "HP:0000211",
    "HP:0009419",
    "HP:0011822",
    "HP:0005616",
    "HP:0011217",
    "HP:0001468",
    "HP:0100633",
    "HP:0040064",
    "HP:0000293",
    "HP:0025264",
    "HP:0006703",
    "HP:0008866",
    "HP:0100059",
    "HP:0012115",
    "HP:0009437",
    "HP:0010669",
    "HP:0004660",
    "HP:0100085",
    "HP:0010606",
    "HP:0008684",
    "HP:0009160",
    "HP:0002683",
    "HP:0009524",
    "HP:0001070",
    "HP:0100774",
    "HP:0012738",
    "HP:0025220",
    "HP:0000132",
    "HP:0009930",
    "HP:0010067",
    "HP:0009145",
    "HP:0012153",
    "HP:0003167",
    "HP:0100097",
    "HP:0000275",
    "HP:0000822",
    "HP:0009239",
    "HP:0500133",
    "HP:0010758",
    "HP:0009234",
    "HP:0009392",
    "HP:0100518",
    "HP:0009211",
    "HP:0009412",
    "HP:0040133",
    "HP:0000453",
    "HP:0009269",
    "HP:0009165",
    "HP:0200085",
    "HP:0003162",
    "HP:0004780",
    "HP:0100043",
    "HP:0009511",
    "HP:0001047",
    "HP:0005964",
    "HP:0012802",
    "HP:0002224",
    "HP:0001770",
    "HP:0009667",
    "HP:0100755",
    "HP:0006375",
    "HP:0100210",
    "HP:0000287",
    "HP:0010469",
    "HP:0008697",
    "HP:0010314",
    "HP:0100430",
    "HP:0004646",
    "HP:0009340",
    "HP:0100463",
    "HP:0012720",
    "HP:0006118",
    "HP:0009157",
    "HP:0001934",
    "HP:0009402",
    "HP:0001117",
    "HP:0009169",
    "HP:0030289",
    "HP:0006986",
    "HP:0000972",
    "HP:0003947",
    "HP:0009236",
    "HP:0004325",
    "HP:0006035",
    "HP:0003231",
    "HP:0004493",
    "HP:0010537",
    "HP:0003196",
    "HP:0100668",
    "HP:0000746",
    "HP:0010541",
    "HP:0003781",
    "HP:0004523",
    "HP:0003073",
    "HP:0006162",
    "HP:0006668",
    "HP:0008069",
    "HP:0006352",
    "HP:0040171",
    "HP:0001882",
    "HP:0000921",
    "HP:0100364",
    "HP:0002094",
    "HP:0001571",
    "HP:0008519",
    "HP:0100371",
    "HP:0040057",
    "HP:0004426",
    "HP:0008249",
    "HP:0009993",
    "HP:0000483",
    "HP:0100265",
    "HP:0011063",
    "HP:0002795",
    "HP:0001362",
    "HP:0009204",
    "HP:0010451",
    "HP:0000122",
    "HP:0000475",
    "HP:0002718",
    "HP:0500153",
    "HP:0003865",
    "HP:0000522",
    "HP:0005867",
    "HP:0009152",
    "HP:0000712",
    "HP:0000478",
    "HP:0100887",
]

    )
)
