import os
import sys
import yaml

from syntype.src.SynTypeConfig import SyntypeConfig

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# =============================================================================
# Configuration File
# =============================================================================

# The settings of syntype ("syntype" section) and of the packages it uses
# (adapter, embedder, llm, logger).
configFile = "../config/config.yaml"

with open(configFile, "r", encoding = "utf-8") as f:
    configurationFile = yaml.safe_load(f)

syntypeConfig = SyntypeConfig.model_validate(configurationFile["syntype"])

# =============================================================================
# Experiment: the words after the model ID on the command line (see
# start.sh) set every setting of the run, a missing word switches its
# setting off. Without such words the configuration file sets them.
# =============================================================================

experiment = syntypeConfig.experiment

def experimentSetting(word : str, configured : bool) -> bool:
    if len(sys.argv) > 2:
        return word in sys.argv[2:]
    return configured

reduceToTestIDs         = experimentSetting("test", experiment.test)
fewShot                 = experimentSetting("few-shot", experiment.few_shot)
chainOfThoughts         = experimentSetting("chain-of-thoughts", experiment.chain_of_thoughts)
addDefinition           = experimentSetting("definition", experiment.definition)
addComment              = experimentSetting("comment", experiment.comment)
addParents              = experimentSetting("parents", experiment.parents)
addChildren             = experimentSetting("children", experiment.children)

forceDrawAdditionalPlots = syntypeConfig.force_draw_additional_plots

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

# The LLMs of syntype.py, classifying one after the other, depend on the
# first command-line argument (see classify.sh):
#   ""         no model: the preparation run of syntype.py, and
#              syntypemerge.py and syntypeeval.py, which combine all models;
#   "all"      all models of llm.model_id (also without any arguments);
#   a model ID only this model.
allModelsArgument = "all"

modelIDs = list(configurationFile["llm"]["model_id"])
if len(sys.argv) > 1 and sys.argv[1] != allModelsArgument:
    modelIDs = []
    if len(sys.argv[1]) > 0 and sys.argv[1][0] != "-":
        modelIDs = [sys.argv[1]]

def modelNameOf(modelID : str) -> str:
    """
    The name of a model in file names and the system column: its ID without
    the organisation, e.g. "medgemma-4b-it" for "google/medgemma-4b-it".
    """
    return modelID[modelID.index("/") + 1:] if "/" in modelID else modelID

similarityColumnPrePrefix   = "similarity_"
similarityColumnPrefix      = f"{similarityColumnPrePrefix}" + "{}_{}"

similarityEvaluationLowerBound = syntypeConfig.similarity_evaluation_lower_bound
similarityEvaluationUperBound  = syntypeConfig.similarity_evaluation_upper_bound
similarityEvaluationParts      = int((similarityEvaluationUperBound - similarityEvaluationLowerBound) * 1000 + 1)

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

# The types the LLM classifies synonyms into (see prompt/).
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
outputFolderNameTransformed         = "transform"
# The second Step is the actual job e.g. generation or classification.
# Third step.
outputFolderNameFormatted           = "format"
# Fourth step.
outputFolderNameMerged              = "merge"
# Fifth step.
outputFolderNameEvaluation          = "evaluate"

def promptLogFileName(modelName : str) -> str:
    """
    The model package's log of the prompts and answers of a model.
    """
    return f"prompts_{testStr}_{configStr}_{chainOfThoughtsStr}_{fewShotStr}_{modelName}.{logFileFormat}"


# The HPO file read by the adapter (adapter.hpo section).
inputFileTransformed        = os.path.join(
    configurationFile["adapter"]["hpo"]["input_folder"],
    configurationFile["adapter"]["hpo"]["input_files"][0]
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

# Similarities of all synonyms to their label (syntype.py ""), for the
# evaluation of test and full runs.
outputFolderNameEmbedded            = "embed"
outputFileEmbedded                  = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEmbedded,
    f"embed.{csvFileFormat}"
)

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

def checkpointFileNameClass(modelName : str) -> str:
    """
    The model package's checkpoint of the histories of a model, in
    llm.prompt_tmp_folder.
    """
    return f"{filesPrefixApproachClass}_{modelName}.tmp"

def outputFileClassFormatted(modelName : str) -> str:
    """
    The formatted classification of a model, merged by syntypemerge.py.
    """
    return os.path.join(dataDir, outputFolderName, outputFolderNameFormatted,
        f"{outputFileNameClassFormattedPrefix}_{modelName}.{csvFileFormat}")

# The formatted classifications of all models of this experiment are read
# from here by syntypemerge.py.
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
outputFileNameClassSimilarityEvaluation   = f"{embeddingPrefix}_similarity_evaluation_" + "{}" + f".{plotFileFormat}"


# =============================================================================

# The examples of the Few-Shot approach. Their synonyms are left out of the
# evaluation (syntypeeval.py; the prompts are in syntype.py), also in runs without examples, so that all
# runs are scored on the same synonyms.
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
        "confidence": "9"
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
        "confidence": "10"
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
        "confidence": "8"
    }
]

# The 634 HPO concepts of the test set ("test").
testIDs = [
    "HP:0000020", "HP:0000046", "HP:0000049", "HP:0000050", "HP:0000062", "HP:0000083",
    "HP:0000104", "HP:0000130", "HP:0000132", "HP:0000144", "HP:0000163", "HP:0000179",
    "HP:0000194", "HP:0000206", "HP:0000214", "HP:0000218", "HP:0000239", "HP:0000269",
    "HP:0000283", "HP:0000298", "HP:0000316", "HP:0000324", "HP:0000336", "HP:0000359",
    "HP:0000378", "HP:0000414", "HP:0000420", "HP:0000452", "HP:0000453", "HP:0000457",
    "HP:0000470", "HP:0000483", "HP:0000486", "HP:0000496", "HP:0000499", "HP:0000517",
    "HP:0000529", "HP:0000544", "HP:0000589", "HP:0000592", "HP:0000599", "HP:0000601",
    "HP:0000602", "HP:0000633", "HP:0000636", "HP:0000653", "HP:0000768", "HP:0000795",
    "HP:0000798", "HP:0000822", "HP:0000834", "HP:0000836", "HP:0000843", "HP:0000851",
    "HP:0000858", "HP:0000875", "HP:0000892", "HP:0000953", "HP:0000961", "HP:0000970",
    "HP:0000978", "HP:0000982", "HP:0001000", "HP:0001012", "HP:0001041", "HP:0001063",
    "HP:0001069", "HP:0001090", "HP:0001106", "HP:0001123", "HP:0001141", "HP:0001155",
    "HP:0001161", "HP:0001166", "HP:0001169", "HP:0001181", "HP:0001195", "HP:0001204",
    "HP:0001250", "HP:0001256", "HP:0001271", "HP:0001288", "HP:0001382", "HP:0001410",
    "HP:0001471", "HP:0001520", "HP:0001522", "HP:0001541", "HP:0001611", "HP:0001623",
    "HP:0001635", "HP:0001649", "HP:0001682", "HP:0001742", "HP:0001763", "HP:0001780",
    "HP:0001783", "HP:0001786", "HP:0001795", "HP:0001805", "HP:0001840", "HP:0001892",
    "HP:0001959", "HP:0001965", "HP:0002034", "HP:0002036", "HP:0002056", "HP:0002059",
    "HP:0002066", "HP:0002069", "HP:0002091", "HP:0002098", "HP:0002105", "HP:0002132",
    "HP:0002149", "HP:0002216", "HP:0002225", "HP:0002242", "HP:0002297", "HP:0002311",
    "HP:0002333", "HP:0002365", "HP:0002367", "HP:0002376", "HP:0002460", "HP:0002511",
    "HP:0002515", "HP:0002557", "HP:0002637", "HP:0002641", "HP:0002647", "HP:0002659",
    "HP:0002663", "HP:0002686", "HP:0002708", "HP:0002758", "HP:0002783", "HP:0002810",
    "HP:0002831", "HP:0002905", "HP:0002914", "HP:0002927", "HP:0002999", "HP:0003002",
    "HP:0003015", "HP:0003022", "HP:0003034", "HP:0003072", "HP:0003073", "HP:0003076",
    "HP:0003099", "HP:0003108", "HP:0003112", "HP:0003113", "HP:0003117", "HP:0003241",
    "HP:0003282", "HP:0003348", "HP:0003367", "HP:0003401", "HP:0003416", "HP:0003521",
    "HP:0003551", "HP:0003658", "HP:0003677", "HP:0003678", "HP:0003717", "HP:0003724",
    "HP:0003733", "HP:0003738", "HP:0003774", "HP:0003795", "HP:0003839", "HP:0003840",
    "HP:0003842", "HP:0003860", "HP:0003879", "HP:0003884", "HP:0003899", "HP:0003904",
    "HP:0003910", "HP:0003920", "HP:0004112", "HP:0004209", "HP:0004224", "HP:0004296",
    "HP:0004334", "HP:0004370", "HP:0004404", "HP:0004426", "HP:0004428", "HP:0004450",
    "HP:0004467", "HP:0004482", "HP:0004488", "HP:0004524", "HP:0004660", "HP:0004938",
    "HP:0004970", "HP:0004975", "HP:0004979", "HP:0004993", "HP:0005005", "HP:0005041",
    "HP:0005059", "HP:0005105", "HP:0005272", "HP:0005285", "HP:0005288", "HP:0005562",
    "HP:0005584", "HP:0005585", "HP:0005768", "HP:0005890", "HP:0005917", "HP:0005924",
    "HP:0006094", "HP:0006118", "HP:0006134", "HP:0006140", "HP:0006162", "HP:0006167",
    "HP:0006257", "HP:0006280", "HP:0006311", "HP:0006313", "HP:0006316", "HP:0006330",
    "HP:0006334", "HP:0006335", "HP:0006353", "HP:0006408", "HP:0006431", "HP:0006456",
    "HP:0006485", "HP:0006488", "HP:0006492", "HP:0006496", "HP:0006597", "HP:0006713",
    "HP:0006753", "HP:0006937", "HP:0006961", "HP:0007058", "HP:0007362", "HP:0007379",
    "HP:0007552", "HP:0007605", "HP:0007626", "HP:0007641", "HP:0007642", "HP:0007687",
    "HP:0007732", "HP:0007835", "HP:0007840", "HP:0008009", "HP:0008067", "HP:0008070",
    "HP:0008083", "HP:0008089", "HP:0008134", "HP:0008283", "HP:0008365", "HP:0008401",
    "HP:0008414", "HP:0008454", "HP:0008465", "HP:0008523", "HP:0008541", "HP:0008665",
    "HP:0008726", "HP:0008771", "HP:0008772", "HP:0008796", "HP:0008812", "HP:0008848",
    "HP:0008935", "HP:0008947", "HP:0008986", "HP:0008998", "HP:0009004", "HP:0009007",
    "HP:0009019", "HP:0009056", "HP:0009058", "HP:0009092", "HP:0009102", "HP:0009105",
    "HP:0009108", "HP:0009124", "HP:0009130", "HP:0009174", "HP:0009202", "HP:0009204",
    "HP:0009209", "HP:0009211", "HP:0009213", "HP:0009223", "HP:0009229", "HP:0009234",
    "HP:0009241", "HP:0009244", "HP:0009246", "HP:0009258", "HP:0009266", "HP:0009267",
    "HP:0009270", "HP:0009273", "HP:0009292", "HP:0009300", "HP:0009317", "HP:0009326",
    "HP:0009341", "HP:0009349", "HP:0009378", "HP:0009380", "HP:0009381", "HP:0009392",
    "HP:0009397", "HP:0009411", "HP:0009415", "HP:0009425", "HP:0009428", "HP:0009436",
    "HP:0009437", "HP:0009444", "HP:0009445", "HP:0009446", "HP:0009450", "HP:0009460",
    "HP:0009469", "HP:0009512", "HP:0009516", "HP:0009519", "HP:0009522", "HP:0009533",
    "HP:0009534", "HP:0009547", "HP:0009548", "HP:0009554", "HP:0009560", "HP:0009562",
    "HP:0009568", "HP:0009575", "HP:0009596", "HP:0009598", "HP:0009599", "HP:0009601",
    "HP:0009613", "HP:0009614", "HP:0009643", "HP:0009644", "HP:0009650", "HP:0009652",
    "HP:0009662", "HP:0009665", "HP:0009669", "HP:0009673", "HP:0009681", "HP:0009683",
    "HP:0009685", "HP:0009696", "HP:0009707", "HP:0009778", "HP:0009779", "HP:0009786",
    "HP:0009802", "HP:0009804", "HP:0009809", "HP:0009823", "HP:0009832", "HP:0009837",
    "HP:0009852", "HP:0009880", "HP:0009933", "HP:0009938", "HP:0009941", "HP:0009945",
    "HP:0009951", "HP:0009963", "HP:0009971", "HP:0009986", "HP:0009987", "HP:0009995",
    "HP:0009997", "HP:0010001", "HP:0010010", "HP:0010015", "HP:0010017", "HP:0010019",
    "HP:0010021", "HP:0010025", "HP:0010031", "HP:0010052", "HP:0010070", "HP:0010074",
    "HP:0010087", "HP:0010091", "HP:0010117", "HP:0010123", "HP:0010127", "HP:0010133",
    "HP:0010135", "HP:0010144", "HP:0010147", "HP:0010150", "HP:0010155", "HP:0010172",
    "HP:0010178", "HP:0010181", "HP:0010194", "HP:0010203", "HP:0010212", "HP:0010220",
    "HP:0010222", "HP:0010224", "HP:0010233", "HP:0010261", "HP:0010270", "HP:0010289",
    "HP:0010318", "HP:0010321", "HP:0010348", "HP:0010365", "HP:0010376", "HP:0010379",
    "HP:0010397", "HP:0010409", "HP:0010411", "HP:0010420", "HP:0010421", "HP:0010430",
    "HP:0010476", "HP:0010484", "HP:0010488", "HP:0010501", "HP:0010511", "HP:0010535",
    "HP:0010560", "HP:0010598", "HP:0010631", "HP:0010668", "HP:0010707", "HP:0010720",
    "HP:0010745", "HP:0010750", "HP:0010751", "HP:0010760", "HP:0010787", "HP:0010830",
    "HP:0010837", "HP:0010881", "HP:0010939", "HP:0011003", "HP:0011029", "HP:0011045",
    "HP:0011057", "HP:0011080", "HP:0011106", "HP:0011108", "HP:0011119", "HP:0011219",
    "HP:0011222", "HP:0011263", "HP:0011282", "HP:0011300", "HP:0011314", "HP:0011363",
    "HP:0011395", "HP:0011450", "HP:0011478", "HP:0011525", "HP:0011799", "HP:0011800",
    "HP:0011947", "HP:0012107", "HP:0012113", "HP:0012116", "HP:0012184", "HP:0012225",
    "HP:0012371", "HP:0012428", "HP:0012432", "HP:0012492", "HP:0012514", "HP:0012532",
    "HP:0012581", "HP:0012583", "HP:0012601", "HP:0012604", "HP:0012637", "HP:0012639",
    "HP:0012699", "HP:0012711", "HP:0012714", "HP:0012718", "HP:0012745", "HP:0012786",
    "HP:0012791", "HP:0012808", "HP:0012809", "HP:0012812", "HP:0012813", "HP:0012819",
    "HP:0012892", "HP:0012893", "HP:0020083", "HP:0025085", "HP:0025205", "HP:0025258",
    "HP:0025262", "HP:0025265", "HP:0025289", "HP:0025553", "HP:0030002", "HP:0030027",
    "HP:0030028", "HP:0030052", "HP:0030148", "HP:0030213", "HP:0030243", "HP:0030284",
    "HP:0030289", "HP:0030299", "HP:0030765", "HP:0031090", "HP:0031091", "HP:0031248",
    "HP:0031259", "HP:0031488", "HP:0031610", "HP:0031921", "HP:0032154", "HP:0032551",
    "HP:0033847", "HP:0040033", "HP:0040070", "HP:0040079", "HP:0040088", "HP:0040095",
    "HP:0045006", "HP:0100007", "HP:0100012", "HP:0100013", "HP:0100022", "HP:0100048",
    "HP:0100050", "HP:0100053", "HP:0100063", "HP:0100064", "HP:0100072", "HP:0100075",
    "HP:0100076", "HP:0100089", "HP:0100093", "HP:0100096", "HP:0100100", "HP:0100111",
    "HP:0100112", "HP:0100137", "HP:0100143", "HP:0100149", "HP:0100150", "HP:0100154",
    "HP:0100182", "HP:0100204", "HP:0100207", "HP:0100211", "HP:0100214", "HP:0100217",
    "HP:0100219", "HP:0100327", "HP:0100379", "HP:0100380", "HP:0100387", "HP:0100388",
    "HP:0100390", "HP:0100394", "HP:0100400", "HP:0100403", "HP:0100408", "HP:0100409",
    "HP:0100415", "HP:0100418", "HP:0100429", "HP:0100430", "HP:0100449", "HP:0100462",
    "HP:0100467", "HP:0100469", "HP:0100476", "HP:0100483", "HP:0100490", "HP:0100502",
    "HP:0100506", "HP:0100519", "HP:0100534", "HP:0100537", "HP:0100550", "HP:0100596",
    "HP:0100604", "HP:0100608", "HP:0100610", "HP:0100614", "HP:0100633", "HP:0100639",
    "HP:0100645", "HP:0100646", "HP:0100651", "HP:0100659", "HP:0100689", "HP:0100720",
    "HP:0100742", "HP:0100783", "HP:0100834", "HP:0100839", "HP:0100903", "HP:0100910",
    "HP:0100918", "HP:0100919", "HP:0100924", "HP:0100933", "HP:0100934", "HP:0100938",
    "HP:0200006", "HP:0410169", "HP:0410214", "HP:0410334", "HP:0430010", "HP:0500011",
    "HP:0500134", "HP:0500143", "HP:0500157", "HP:3000053"
]
