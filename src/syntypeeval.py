import numpy                as np
import matplotlib.pyplot    as plt
import seaborn              as sns
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.lines import Line2D
import sys
import shlex
import math

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from utils      import *
from config     import *

from logger     import Logger

from collections import Counter

# The logger takes the "logger" section of the configuration file; the
# command line of the run is its first line, since the log file is named
# after the start of the run only.
l             = Logger(config = configFile)
l.log(f"Command: python3 {shlex.join(sys.argv)}")

l.printHeader("Evaluating the Results of Synonym Classification")

# The logger logs the start, end and duration of the step.
with l.step("Evaluating the classifications"):
    # Only proceed if formatted input data exists
    exitIfFileNotExist(inputFileClassEvaluation)

    classifiedComplete = readCSV(inputFileClassEvaluation)

    # The similarities of the synonyms to their label (the preparation run,
    # syntype.py ""). Similarity columns of classifications made before the
    # embeddings got their own file are replaced.
    if isFile(outputFileEmbedded):
        similarities = readCSV(outputFileEmbedded)
        keys         = [hpoidColumn, classColumn, contentColumn]
        for data in [classifiedComplete, similarities]:
            data[keys] = data[keys].astype(str)

        classifiedComplete = classifiedComplete.drop(columns = [
            c for c in classifiedComplete.columns
            if c.startswith(similarityColumnPrePrefix)
        ]).merge(similarities, on = keys, how = "left")
    else:
        l.log(f"No similarities found ({os.path.basename(outputFileEmbedded)}); " \
            "run syntype.py \"\" for the similarity plots.")

    ontologies = [
        str(s).split(":", 1)[0] for s in classifiedComplete[hpoidColumn].to_list()
    ]

    counts = Counter(ontologies)

    l.log(f"{len(counts.keys())} ontologies found.")

    for key in counts.keys():
        l.log(f"Found {counts[key]} entries in HPO for ontology '{key}'.")

    systems = list(set(classifiedComplete[systemColumn].tolist()))
    systems = sorted(systems)

    l.log(f"Found Systems: {applyFormat(systems)}")
    l.log(f"Classified Synonyms: {len(classifiedComplete.index)} " \
        f"(~{int(len(classifiedComplete.index) / len(systems))} per system)")
    classifiedComplete = classifiedComplete[classifiedComplete[systemColumn] != ""]

    # Change Datatype to String and lower case everything.
    classifiedComplete[classColumn]   = classifiedComplete[classColumn ].str.lower()
    classifiedComplete[answerColumn]  = classifiedComplete[answerColumn].str.lower()
    classifiedComplete[typeColumn]    = classifiedComplete[typeColumn  ].str.lower()

    # Synonyms without a source type are set to "expert" (synonymTypeOf() in
    # utils.py does this already; "direct" remains in classifications made
    # before that).
    classifiedComplete[typeColumn]    = classifiedComplete[typeColumn].replace(
        [np.nan, directSynonymType], expertSynonymType.lower()
    )

    # ------------------------------------------------------------------------------
    # The synonyms are classified into the source types synonymTypes (expert,
    # layperson); only synonyms with one of these as gold type are evaluated.
    # Other gold types (abbreviation, uk, plural, ...) describe the form of a
    # synonym, not whether it is an expert or a layperson term.
    # ------------------------------------------------------------------------------

    classifiedComplete = classifiedComplete[classifiedComplete[classColumn].isin(synonymClasses)]

    goldSystem = classifiedComplete[classifiedComplete[systemColumn] == systems[0]]
    excluded   = goldSystem[~goldSystem[typeColumn].isin(synonymTypes)][typeColumn].value_counts()
    l.log(f"Evaluating the synonyms with gold type {applyFormat(synonymTypes)}; " \
        f"left out: {', '.join(f'{count} {t}' for t, count in excluded.items()) or 'none'}.")

    classifiedComplete = classifiedComplete[
        classifiedComplete[typeColumn].isin(synonymTypes)
    ].copy().reset_index(drop = True)

    # The synonyms of the Few-Shot examples are given with their answer in the
    # prompts, so they are not evaluated.
    exampleSynonyms = classifiedComplete[[hpoidColumn, contentColumn]].apply(tuple, axis = 1).isin(
        [(example[hpoidColumn], example["synonym"]) for example in fewShotExamples])
    l.log(f"Leaving out the {len(fewShotExamples)} synonyms of the Few-Shot examples " \
        f"({exampleSynonyms.sum()} rows).")
    classifiedComplete = classifiedComplete[~exampleSynonyms].copy().reset_index(drop = True)

    colors = plt.cm.tab10(range(len(systems) + 1))
    color_map = {
        system: plt.cm.tab10(i % 10)
        for i, system in enumerate(systems)
    }

    def f1Score(predicted : pd.Series, gold : pd.Series, sourceType : str) -> float:
        """
        F1 score of the source type sourceType for a prediction (np.nan if
        nothing is predicted as sourceType).
        """
        TP = sum(predicted & (gold == sourceType))
        FP = sum(predicted & (gold != sourceType))
        FN = sum(~predicted & (gold == sourceType))

        if TP + FP == 0:
            return np.nan
        return (2 * TP) / (2 * TP + FP + FN)

    # ------------------------------------------------------------------------------
    # Similarity of the synonyms to their label per gold type, and the threshold
    # that best separates layperson synonyms (below) from expert synonyms (above).
    # ------------------------------------------------------------------------------

    if reduceToTestIDs or forceDrawAdditionalPlots:
        classified = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy().reset_index(drop = True)
        for embeddingModel in embeddingModels.keys():
            for similarityMetric in similarityMetrics:
                column = similarityColumnPrefix.format(embeddingModel, similarityMetric)

                outputFileClassEmbeddingEval              = os.path.join(
                    dataDir,
                    outputFolderName,
                    outputFolderNameClassEmbedding,
                    outputFileNameClassEmbeddingEvaluation.format(column)
                )

                outputFileClassSimilarityEval              = os.path.join(
                    dataDir,
                    outputFolderName,
                    outputFolderNameClassEmbedding,
                    outputFileNameClassSimilarityEvaluation.format(column)
                )

                if column in classified.columns and (not isFile(outputFileClassSimilarityEval) or forceDrawAdditionalPlots):
                    plt.figure(figsize=(8, 5))

                    palette = {
                            expertSynonymType    : colors[0],
                            laypersonSynonymType : colors[1]
                        }

                    # Draw density curves for each source type
                    sns.kdeplot(
                        data=classified,
                        x=column,
                        palette=palette,
                        hue=classified[typeColumn],
                        common_norm=False,   # keeps each type independent
                        fill=False           # set True if you want filled curves
                    )

                    plt.xlabel("Similarity")
                    plt.ylabel("Density")
                    plt.title("Density Curves of Similarity by Source Type")
                    plt.yticks(np.linspace(0, 1, 6))
                    plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 9))
                    plt.xlim(similarityEvaluationLowerBound, similarityEvaluationUperBound)

                    handles = [
                        Line2D([0], [0], color=color, lw=2, label=cls)
                        for cls, color in palette.items()
                    ]

                    plt.legend(handles=handles, title="Source Type")

                    plt.grid(axis = "both")

                    plt.tight_layout()
                    plt.savefig(outputFileClassSimilarityEval, dpi = 300, bbox_inches = "tight")


                if column in classified.columns and (not isFile(outputFileClassEmbeddingEval) or forceDrawAdditionalPlots):
                    l.log(f"Plotting {similarityMetric} for Model {embeddingModel}...")

                    x1 = []
                    y1 = []
                    x2 = []
                    y2 = []

                    with newProgress() as progress:

                        task = newTask(progress, similarityEvaluationParts, "Evaluating")

                        for i in np.linspace(similarityEvaluationLowerBound,
                                            similarityEvaluationUperBound,
                                            similarityEvaluationParts):
                            x1.append(i)
                            x2.append(i)

                            y1.append(f1Score(classified[column] <= i,
                                classified[typeColumn], laypersonSynonymType))
                            y2.append(f1Score(classified[column] >= i,
                                classified[typeColumn], expertSynonymType))

                            progress.update(task, advance = 1)

                        progress.refresh()

                    # Create the bar plot
                    plt.figure()
                    plt.plot(x1, y1, label = "F1 Score (Layperson Threshold)",
                        color = "red")
                    plt.plot(x2, y2, label = "F1 Score (Expert Threshold)",
                        color = "blue")

                    x1 = np.array(x1)
                    y1 = np.array(y1)
                    x2 = np.array(x2)
                    y2 = np.array(y2)

                    roundFactor = float(math.pow(10, len(str(similarityEvaluationParts)) - 1))

                    if not np.all(np.isnan(y1)):
                        max_index = np.nanargmax(y1)
                        max_x = math.ceil(roundFactor * x1[max_index]) / roundFactor
                        max_y = math.ceil(roundFactor * y1[max_index]) / roundFactor

                        c = sum(classified[column] <= max_x)
                        p = 100 * float(math.floor(roundFactor * c / len(classified.index))) / roundFactor


                        plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "red", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                        l.log(f"Suggested Threshold for Type {quote(laypersonSynonymType)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)} with F1 of {max_y} classifying {str(p)}% of the data." )

                    if not np.all(np.isnan(y2)):
                        max_index = np.nanargmax(y2)
                        max_x = math.ceil(roundFactor * x2[max_index]) / roundFactor
                        max_y = math.ceil(roundFactor * y2[max_index]) / roundFactor

                        c = sum(classified[column] >= max_x)
                        p = 100 * float(math.floor(roundFactor * c / len(classified.index)))  / roundFactor

                        plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "blue", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                        l.log(f"Suggested Threshold for Type {quote(expertSynonymType)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)} with F1 of {max_y} classifying {str(p)}% of the data." )

                    plt.xlabel("Threshold")
                    plt.ylabel("F1 Score")
                    plt.title(f"Normalized {similarityMetric.capitalize()}-Similarity for Model {quote(embeddingModel)}")
                    plt.grid(axis = "both")

                    plt.yticks(np.linspace(0, 1.0, 11))
                    plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 17))

                    plt.legend(loc = "lower left", ncol = 1)
                    plt.savefig(outputFileClassEmbeddingEval, dpi = 300, bbox_inches = "tight")

    # ------------------------------------------------------------------------------
    # Gold counts.
    # ------------------------------------------------------------------------------

    classified = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy().drop([answerColumn, systemColumn], axis = 1).drop_duplicates().reset_index(drop = True)
    types = Counter(classified[typeColumn])

    # Prepare data for plotting
    labels = list(types.keys())
    values = list(types.values())

    # Create the bar plot
    plt.figure()
    bars = plt.bar(labels, values, color = colors)

    # Add counts on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha = 'center',
            va = 'bottom'
        )

    plt.xlabel("Source Type")
    plt.ylabel("Count")
    plt.title("Count of Source Types")
    plt.grid(axis = "both")
    plt.savefig(outputFileClassGoldCounts, dpi = 300, bbox_inches = "tight")

    # ------------------------------------------------------------------------------
    # Classification counts per system.
    # ------------------------------------------------------------------------------

    counts = classifiedComplete.groupby([answerColumn, systemColumn]).size().unstack(fill_value = 0)
    # Create plot
    ax = counts.plot(kind = "bar", figsize = (3 * len(systems), 4), width = 0.8)

    # Add value labels above bars
    for container in ax.containers:
        ax.bar_label(container, padding = 3)

    plt.xlabel("Classified Source Type")
    plt.ylabel("Count")
    plt.title("Count of Classified Source Types")
    plt.xticks(rotation = 0)
    plt.grid(axis = "both")
    plt.savefig(outputFileClassAnswerCounts, dpi = 300, bbox_inches = "tight")

    # ------------------------------------------------------------------------------
    # Precision, recall and F1 per system and source type.
    # ------------------------------------------------------------------------------

    def ontologySubplot(data : pd.DataFrame = None, startString : str = "", outputFile : str = "", title : str = "") -> None:
        classified = data[data[hpoidColumn].str.startswith(startString, na=False)].reset_index(drop = True).copy()

        # E.g. the test set has only HPO concepts.
        if len(classified.index) == 0:
            l.log(f"No synonyms of concepts starting with {quote(startString)}; skipping the plot {quote(title)}.")
            return

        result = {}

        systems = sorted(set(data[systemColumn].tolist()))
        for system in systems:
            systemData = classified[classified[systemColumn] == system]

            # Undefined answers count as wrong for every source type. A type
            # that is never predicted (or not in the gold data) scores 0.
            precision, recall, f1, _ = precision_recall_fscore_support(
                systemData[typeColumn], systemData[answerColumn],
                labels = synonymTypes, average = None, zero_division = 0
            )

            result[system] = {
                sourceType : {
                    precisionLabel : precision[i],
                    recallLabel    : recall[i],
                    f1ScoreLabel   : f1[i]
                }
                for i, sourceType in enumerate(synonymTypes)
            }

            l.log(f"{system} ({quote(startString or 'all')}): " + ", ".join(
                f"{t}: P {result[system][t][precisionLabel]:.3f}, " \
                f"R {result[system][t][recallLabel]:.3f}, " \
                f"F1 {result[system][t][f1ScoreLabel]:.3f}"
                for t in synonymTypes))

        metrics = [f1ScoreLabel, recallLabel, precisionLabel]
        classes = list(synonymTypes)

        x = np.arange(len(systems)) * 0.1 * len(systems)
        barWidth = 0.2

        fig, axes = plt.subplots(
            nrows = len(metrics),
            ncols = len(classes),
            figsize = (3 * len(metrics), 2 * len(classes)),
            sharey = True,
            squeeze = False
        )

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[i])
            for i in systems
        ]

        fig.legend(
            handles,
            systems,
            loc = "lower center",
            ncol = int(len(systems) / 2 if len(systems) >= 2 else 1 ),
            frameon = False
        )

        for i, cls in enumerate(classes):
            for j, metric in enumerate(metrics):
                ax = axes[j, i]
                ax.grid(True, axis="both")
                values = [result[system][cls][metric] for system in systems]

                for k, system in enumerate(systems):
                    bars = ax.bar(
                        x[k],
                        values[k],
                        barWidth,
                        color = color_map[system]
                    )

                    for bar in bars:
                        h = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.01,              # small vertical offset
                            f"{h:.2f}",
                            ha = "center",
                            va = "bottom",
                            fontsize = 7
                        )

                ax.set_ylim(0, 1)
                ax.set_xticks(x)
                ax.set_xticklabels([""] * len(systems))

                if i == 0:
                    ax.set_ylabel(metric.capitalize())
                if j == 0:
                    ax.set_title(cls)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect = [0, 0.05, 1, 1])
        plt.savefig(outputFile, dpi = 300, bbox_inches = "tight")

    # Plot evaluation on different ontologies.
    ontologySubplot(classifiedComplete, "",        outputFileClassRecallPrecisionF1,     "Source Type Classification Performance of LLM Systems")
    ontologySubplot(classifiedComplete, "HP:",     outputFileClassEvaluationExactHPO,    "Source Type Classification Performance of LLM Systems (HPO only)")
    ontologySubplot(classifiedComplete, "UBERON:", outputFileClassEvaluationExactUBERON, "Source Type Classification Performance of LLM Systems (UBERON only)")
    ontologySubplot(classifiedComplete, "GO:",     outputFileClassEvaluationExactGO,     "Source Type Classification Performance of LLM Systems (GO only)")
    ontologySubplot(classifiedComplete, "CHEBI:",  outputFileClassEvaluationExactCHEBI,  "Source Type Classification Performance of LLM Systems (CHEBI only)")
