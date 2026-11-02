import numpy                as np
import matplotlib.pyplot    as plt
import seaborn              as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from matplotlib.lines import Line2D
import sys
import time
import math

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from utils      import *
from config     import *

from collections import Counter

printHeader("Evaluating the Results of Synonym Classification")
startTime = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassEvaluation)

classifiedComplete = readCSV(inputFileClassEvaluation)
prefixes = ["nan", "RO", "core#site"]

# Keep rows that do NOT start with any prefix in the list
classifiedComplete = classifiedComplete[~classifiedComplete[hpoidColumn].str.startswith(tuple(prefixes))]
classifiedComplete = classifiedComplete[~classifiedComplete[hpoidColumn].isna()]
classifiedComplete.loc[classifiedComplete[typeColumn] == directSynonymType, typeColumn] = expertSynonymType
classifiedComplete = classifiedComplete[classifiedComplete[typeColumn].isin(synonymTypes)].reset_index(drop = True)

for index, row in classifiedComplete.iterrows():
    if row[typeColumn] == directSynonymType:
        classifiedComplete.loc[index, typeColumn] = expertSynonymType




ontologies = [
    str(s).split(":", 1)[0] for s in classifiedComplete[hpoidColumn].to_list()
]

counts = Counter(ontologies)

log(f"{len(counts.keys())} ontologies found.")

for key in counts.keys():
    log(f"Found {counts[key]} entries in HPO for ontology '{key}'.")

systems = list(set(classifiedComplete[systemColumn].tolist()))
systems = sorted(systems)

log(f"Found Systems: {applyFormat(systems)}")
log(f"Classified Synonyms: {len(classifiedComplete.index)} " \
    f"(~{int(len(classifiedComplete.index) / len(systems))} per system)")
classifiedComplete = classifiedComplete[classifiedComplete[systemColumn] != ""]

# Change Datatype to String and lower case everything. 
classifiedComplete[classColumn]   = classifiedComplete[classColumn ].str.lower()
classifiedComplete[answerColumn]  = classifiedComplete[answerColumn].str.lower()
classifiedComplete[typeColumn]    = classifiedComplete[typeColumn  ].str.lower()

# Synonyms without a source type are set to "expert".
classifiedComplete[typeColumn]    = classifiedComplete[typeColumn].replace(
    np.nan, expertSynonymType.lower()
)

colors = plt.cm.tab10(range(len(systems) + 1))
color_map = {
    system: plt.cm.tab10(i % 10)
    for i, system in enumerate(systems)
}




"""
# Ploting Similarity distributions and threshold calculation.
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

            if column in classified.columns and (not os.path.isfile(outputFileClassSimilarityEval) or forceDrawAdditionalPlots):
                plt.figure(figsize=(8, 5))

                palette = {
                        expertSynonymType : colors[0],
                        laypersonSynonymType : colors[1]
                    }

                # Draw density curves for each class
                sns.kdeplot(
                    data=classified,
                    x=column,
                    palette=palette,
                    hue=classified[typeColumn],
                    common_norm=False,   # keeps each class independent
                    fill=False           # set True if you want filled curves
                )

                plt.xlabel("Similarity")
                plt.ylabel("Density")
                plt.title("Density Curves of Similarity by Class")
                plt.yticks(np.linspace(0, 1, 6))
                plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 9))
                plt.xlim(similarityEvaluationLowerBound, similarityEvaluationUperBound)
                
                handles = [
                    Line2D([0], [0], color=color, lw=2, label=cls)
                    for cls, color in palette.items()
                ]

                plt.legend(handles=handles, title="Class")

                plt.grid(axis = "both")

                plt.tight_layout()
                plt.savefig(outputFileClassSimilarityEval, dpi = 300, bbox_inches = "tight")


            if column in classified.columns and (not os.path.isfile(outputFileClassEmbeddingEval) or forceDrawAdditionalPlots):
                log(f"Plotting {similarityMetric} for Model {embeddingModel}...")

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

                        TP = sum((
                            classified[column] <= i
                                ) & (
                            classified[typeColumn] == laypersonSynonymType
                        ))
                        FP = sum((
                            classified[column] <= i
                                ) & (
                            classified[typeColumn] != laypersonSynonymType
                        ))

                        if TP + FP > 0:
                            y1.append((2 * TP) / (2 * TP + FP))
                        else:
                            y1.append(np.nan)

                        TP = sum((
                            classified[column] >= i
                                ) & (
                            classified[typeColumn] == expertSynonymType
                        ))
                        FP = sum((
                            classified[column] >= i
                                ) & (
                            classified[typeColumn] != expertSynonymType
                        ))

                        if TP + FP > 0:
                            y2.append((2 * TP) / (2 * TP + FP))
                        else:
                            y2.append(np.nan)

                        progress.update(task, advance = 1)
        
                    progress.refresh()

                # Create the bar plot
                plt.figure()
                plt.plot(x1, y1, label = "F1 Score (Related Threshold)", 
                    color = "red")
                plt.plot(x2, y2, label = "F1 Score (Expert Threshold)",   
                    color = "blue")
                
                x1 = np.array(x1)
                y1 = np.array(y1)
                x2 = np.array(x2)
                y2 = np.array(y2)

                x1 = x1[y1 < 1]
                y1 = y1[y1 < 1]

                x2 = x2[y2 < 1]
                y2 = y2[y2 < 1]

                roundFactor = float(math.pow(10, len(str(similarityEvaluationParts)) - 1))

                if not np.all(np.isnan(y1)):
                    max_index = np.nanargmax(y1)
                    max_x = math.ceil(roundFactor * x1[max_index]) / roundFactor
                    max_y = math.ceil(roundFactor * y1[max_index]) / roundFactor

                    c = sum(classified[column] <= max_x)
                    p = 100 * float(math.floor(roundFactor * c / len(classified.index))) / roundFactor


                    plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "red", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                    log(f"Suggested Threshold for Class {quote(relatedSynonymClass)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)} with F1 of {max_y} classifying {str(p)}% of the data." )

                if not np.all(np.isnan(y2)):
                    max_index = np.nanargmax(y2)
                    max_x = math.ceil(roundFactor * x2[max_index]) / roundFactor
                    max_y = math.ceil(roundFactor * y2[max_index]) / roundFactor

                    c = sum(classified[column] >= max_x)
                    p = 100 * float(math.floor(roundFactor * c / len(classified.index)))  / roundFactor

                    plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "blue", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                    log(f"Suggested Threshold for Class {quote(expertSynonymType)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)} with F1 of {max_y} classifying {str(p)}% of the data." )

                plt.xlabel("Threshold")
                plt.ylabel("F1 Score")
                plt.title(f"Normalized {similarityMetric.capitalize()}-Similarity for Model {quote(embeddingModel)}")
                plt.grid(axis = "both")

                plt.yticks(np.linspace(0, 1.0, 11))
                plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 17))

                plt.legend(loc = "lower left", ncol = 1)
                plt.savefig(outputFileClassEmbeddingEval, dpi = 300, bbox_inches = "tight")
"""

threshold = 0.442
column    = similarityColumnPrefix.format(sapBERT,         cosineSimilarity)

data = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy()
data = data[data[typeColumn].isin(synonymTypes)].copy()

data["y_pred"] = data[column].apply(
    lambda x: laypersonSynonymType if x <= threshold else expertSynonymType
)

y_true = data[typeColumn]
y_pred = data["y_pred"]

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0,
    labels=[laypersonSynonymType, expertSynonymType]
)

macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

log(f"Overall Accuracy: {accuracy:.4f}")
log(f"Macro Precision:  {macro_p:.4f}")
log(f"Macro Recall:     {macro_r:.4f}")
log(f"Macro F1:         {macro_f1:.4f}")

for i, label in enumerate([laypersonSynonymType, expertSynonymType]):
    log(f"[{label}] Precision: {precision[i]:.4f}, "
        f"Recall: {recall[i]:.4f}, "
        f"F1: {f1[i]:.4f}, "
        f"Support: {support[i]}")



classified_backup = classifiedComplete.copy()
# Ploting Similarity distributions and threshold calculation.
if reduceToTestIDs or forceDrawAdditionalPlots:
    classified = readCSV(outputFileTransformed)

    classified = classified[~classified[hpoidColumn].isna()]

    for index, row in classified.iterrows():
        if row[typeColumn] == directSynonymType:
            classified.loc[index, typeColumn] = expertSynonymType

    #classified = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy().reset_index(drop = True)

    for embeddingModel in embeddingModels.keys():
        for similarityMetric in similarityMetrics:
            column = similarityColumnPrefix.format(embeddingModel, similarityMetric)

            total      = len(classified[classified[typeColumn].isin(synonymTypes)])
            thresholds = np.linspace(similarityEvaluationLowerBound,
                                    similarityEvaluationUperBound,
                                    similarityEvaluationParts)

            accuracies = []

            for i in thresholds:
                # <=i → layperson, >i → expert
                predictedLayperson = classified[column] <= i
                correct = (
                    ( predictedLayperson & (classified[typeColumn] == laypersonSynonymType)) |
                    (~predictedLayperson & (classified[typeColumn] == expertSynonymType))
                )
                accuracies.append(correct.sum() / total if total > 0 else np.nan)

            accuracies = np.array(accuracies)

            best_idx       = np.nanargmax(accuracies)
            best_threshold = thresholds[best_idx]
            best_accuracy  = accuracies[best_idx]

            log(f"[{embeddingModel} | {similarityMetric}] "
                f"Best Threshold: {best_threshold:.4f} "
                f"→ Accuracy: {best_accuracy:.4f}")

            # --- Precision, Recall, F1 at best threshold ---
            classified_synonyms = classified[classified[typeColumn].isin(synonymTypes)].copy()

            y_true = classified_synonyms[typeColumn]
            y_pred = (classified_synonyms[column].apply(
                lambda x: laypersonSynonymType if x <= best_threshold else expertSynonymType
            ))

            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0,
                labels=[laypersonSynonymType, expertSynonymType]
            )

            for i, label in enumerate([laypersonSynonymType, expertSynonymType]):
                log(f"[{embeddingModel} | {similarityMetric} | {label}] "
                    f"Precision: {precision[i]:.4f}, "
                    f"Recall: {recall[i]:.4f}, "
                    f"F1: {f1[i]:.4f}, "
                    f"Support: {support[i]}")

            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            log(f"[{embeddingModel} | {similarityMetric} | macro] "
                f"Precision: {macro_p:.4f}, "
                f"Recall: {macro_r:.4f}, "
                f"F1: {macro_f1:.4f}")

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

            if column in classified.columns and (not os.path.isfile(outputFileClassSimilarityEval) or forceDrawAdditionalPlots):
                plt.figure(figsize=(8, 5))

                palette = {
                        expertSynonymType : colors[0],
                        laypersonSynonymType : colors[1]
                    }

                # Draw density curves for each class
                sns.kdeplot(
                    data=classified,
                    x=column,
                    palette=palette,
                    hue=classified[typeColumn],
                    common_norm=False,   # keeps each class independent
                    fill=False           # set True if you want filled curves
                )

                plt.xlabel("Distance")
                plt.ylabel("Density")
                plt.title("Density Curves of Distance by Class")
                plt.yticks(np.linspace(0, 1, 6))
                plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 9))
                plt.xlim(similarityEvaluationLowerBound, similarityEvaluationUperBound)
                
                handles = [
                    Line2D([0], [0], color=color, lw=2, label=cls)
                    for cls, color in palette.items()
                ]

                plt.legend(handles=handles, title="Class")

                plt.grid(axis = "both")

                plt.tight_layout()
                plt.savefig(outputFileClassSimilarityEval, dpi = 300, bbox_inches = "tight")


            if column in classified.columns and (not os.path.isfile(outputFileClassEmbeddingEval) or forceDrawAdditionalPlots):
                log(f"Plotting {similarityMetric} for Model {embeddingModel}...")

                x1 = []
                y1 = []
                x2 = []
                y2 = []

                total = len(classified)
                
                with newProgress() as progress:
                    
                    task = newTask(progress, similarityEvaluationParts, "Evaluating")

                    for i in np.linspace(similarityEvaluationLowerBound, 
                                        similarityEvaluationUperBound, 
                                        similarityEvaluationParts):
                        x1.append(i)
                        x2.append(i)

                        # Accuracy for layperson threshold (<=i means layperson, >i means expert)
                        predictedLayperson  = classified[column] <= i
                        predictedExpert     = ~predictedLayperson
                        correct1 = (
                            (predictedLayperson & (classified[typeColumn] == laypersonSynonymType)) |
                            (predictedExpert    & (classified[typeColumn] == expertSynonymType))
                        )
                        y1.append(correct1.sum() / total if total > 0 else np.nan)

                        # Accuracy for expert threshold (>=i means expert, <i means layperson)
                        predictedExpert2    = classified[column] >= i
                        predictedLayperson2 = ~predictedExpert2
                        correct2 = (
                            (predictedExpert2    & (classified[typeColumn] == expertSynonymType)) |
                            (predictedLayperson2 & (classified[typeColumn] == laypersonSynonymType))
                        )
                        y2.append(correct2.sum() / total if total > 0 else np.nan)

                        progress.update(task, advance = 1)
        
                    progress.refresh()

                # Create the plot
                plt.figure()
                plt.plot(x1, y1, label = "Accuracy (Layperson Threshold)", 
                    color = "red")
                plt.plot(x2, y2, label = "Accuracy (Expert Threshold)",   
                    color = "blue")
                
                x1 = np.array(x1)
                y1 = np.array(y1)
                x2 = np.array(x2)
                y2 = np.array(y2)

                x1 = x1[y1 < 1]
                y1 = y1[y1 < 1]

                x2 = x2[y2 < 1]
                y2 = y2[y2 < 1]

                roundFactor = float(math.pow(10, len(str(similarityEvaluationParts)) - 1))

                if not np.all(np.isnan(y1)):
                    max_index = np.nanargmax(y1)
                    max_x = math.ceil(roundFactor * x1[max_index]) / roundFactor
                    max_y = math.ceil(roundFactor * y1[max_index]) / roundFactor

                    c = sum(classified[column] <= max_x)
                    p = 100 * float(math.floor(roundFactor * c / len(classified.index))) / roundFactor

                    plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "red", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                    log(f"Suggested Threshold for Class {quote(laypersonSynonymType)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)} with Accuracy of {max_y} classifying {str(p)}% of the data.")

                if not np.all(np.isnan(y2)):
                    max_index = np.nanargmax(y2)
                    max_x = math.ceil(roundFactor * x2[max_index]) / roundFactor
                    max_y = math.ceil(roundFactor * y2[max_index]) / roundFactor

                    c = sum(classified[column] >= max_x)
                    p = 100 * float(math.floor(roundFactor * c / len(classified.index)))  / roundFactor

                    plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "blue", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                    log(f"Suggested Threshold for Class {quote(expertSynonymType)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)} with Accuracy of {max_y} classifying {str(p)}% of the data.")

                plt.xlabel("Threshold")
                plt.ylabel("Accuracy")
                plt.title(f"Normalized {similarityMetric.capitalize()}-Distance for Model {quote(embeddingModel)}")
                plt.grid(axis = "both")

                plt.yticks(np.linspace(0, 1.0, 11))
                plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 17))

                plt.legend(loc = "lower left", ncol = 1)
                plt.savefig(outputFileClassEmbeddingEval, dpi = 300, bbox_inches = "tight")



data = classifiedComplete[classifiedComplete[typeColumn].isin(synonymTypes)].copy()

data["correct"] = data[answerColumn] == data[typeColumn]

llm_stats = data.groupby([hpoidColumn, contentColumn, typeColumn]).agg(
    llm_count     = (systemColumn, "nunique"),
    correct_count = ("correct",    "sum")
).reset_index()

log(f"Total unique synonyms found: {len(llm_stats)}")

for synonymType in sorted(llm_stats[typeColumn].unique()):
    subset_type = llm_stats[llm_stats[typeColumn] == synonymType]
    total_type  = len(subset_type)
    p_type      = 100 * total_type / len(llm_stats)

    log(f"\n  Type '{synonymType}': {total_type} synonyms ({p_type:.2f}% of all)")

    for k in sorted(subset_type["llm_count"].unique()):
        subset = subset_type[subset_type["llm_count"] == k]
        total  = len(subset)
        p      = 100 * total / total_type

        log(f"    {k} LLM(s): {total} synonyms ({p:.2f}%)")

        for c in range(k + 1):
            count = (subset["correct_count"] == c).sum()
            p_c   = 100 * count / total if total > 0 else 0
            log(f"      {c}/{k} correct: {count} synonyms ({p_c:.2f}%)")




# Plot Gold Counts
classified = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy().drop([answerColumn, systemColumn], axis = 1).drop_duplicates().reset_index(drop = True)
classes = Counter(classified[typeColumn])

# Prepare data for plotting
labels = list(classes.keys())
values = list(classes.values())

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
plt.show()
plt.savefig(outputFileClassGoldCounts, dpi = 300, bbox_inches = "tight")






# Plot Classification Counts
classifiedComplete = classifiedComplete[classifiedComplete[typeColumn].isin(synonymTypes)].copy().reset_index(drop = True)

# Count occurrences per system and classification
classified = classifiedComplete
counts = classifiedComplete.groupby([answerColumn, systemColumn]).size().unstack(fill_value = 0)
# Create plot
ax = counts.plot(kind = "bar", figsize = (3 * len(systems), 4), width = 0.8)

# Add value labels above bars
for container in ax.containers:
    ax.bar_label(container, padding = 3)

plt.xlabel("Classified Synonym Type")
plt.ylabel("Count")
plt.title("Count of Classified Synonym Type")
plt.xticks(rotation = 0)
plt.grid(axis = "both")
plt.show()
plt.savefig(outputFileClassAnswerCounts, dpi = 300, bbox_inches = "tight")





def ontologySubplot(data : pd.DataFrame = None, startString : str = "", outputFile : str = "", title : str = "") -> None:
    classified = data[data[hpoidColumn].str.startswith(startString, na=False)].reset_index(drop = True).copy()

    result = {}

    systems = list(set(data[systemColumn].tolist()))
    for system in systems:
        systemData = classified[classified[systemColumn] == system]

        systemResults = {}

        if systemData is not None and len(systemData.index) > 0:
            for synonymClasse in synonymTypes:
                
                systemClassResults = {
                    precisionLabel  : 0,
                    recallLabel     : 0,
                    f1ScoreLabel    : 0,
                    accuracyLabel   : 0   # NEW
                }

                if len(systemData[systemData[answerColumn] == 
                    synonymClasse].index) > 0:

                    tp = len(systemData[
                            (systemData[typeColumn] == synonymClasse) & 
                            (systemData[answerColumn] == synonymClasse)
                        ].index)
                    fp = len(systemData[
                            (systemData[typeColumn] != synonymClasse) & 
                            (systemData[answerColumn] == synonymClasse)
                        ].index)
                    fn = len(systemData[
                            (systemData[typeColumn] == synonymClasse) & 
                            (systemData[answerColumn] != synonymClasse)
                        ].index)
                    tn = len(systemData[
                            (systemData[typeColumn] != synonymClasse) & 
                            (systemData[answerColumn] != synonymClasse)
                        ].index)
                    t = len(systemData[
                        (systemData[typeColumn] == systemData[answerColumn])
                    ])

                    total = tp + fp + fn + tn

                    systemClassResults[precisionLabel] = tp / (tp + fp) if (tp + fp) > 0 else 0
                    systemClassResults[recallLabel]    = tp / (tp + fn) if (tp + fn) > 0 else 0

                    p = systemClassResults[precisionLabel]
                    r = systemClassResults[recallLabel]
                    systemClassResults[f1ScoreLabel]   = 2 * p * r / (p + r) if (p + r) > 0 else 0

                    # NEW: accuracy = (TP + TN) / total
                    systemClassResults[accuracyLabel]  = t / total if total > 0 else 0

                else:
                    systemClassResults[f1ScoreLabel]    = 0
                    systemClassResults[recallLabel]     = 0
                    systemClassResults[precisionLabel]  = 0
                    systemClassResults[accuracyLabel]   = 0   # NEW

                systemResults[synonymClasse] = systemClassResults

        result[system] = systemResults

    metrics = [f1ScoreLabel, recallLabel, precisionLabel, accuracyLabel]   # NEW: added accuracyLabel
    systems = list(result.keys())
    classes = list(next(iter(result.values())).keys())

    x = np.arange(len(systems)) * 0.1 * len(systems)
    barWidth = 0.2

    fig, axes = plt.subplots(
        nrows = len(metrics),
        ncols = len(classes),
        figsize = (3 * len(metrics), 2 * len(classes)),   # width scales with metrics count
        sharey = True
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[i])
        for i in systems
    ]

    fig.legend(
        handles,
        systems,
        loc = "lower center",
        ncol = int(len(systems) / 2 if len(systems) >= 2 else 1),
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
                        h + 0.01,
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
ontologySubplot(classifiedComplete, "",        outputFileClassRecallPrecisionF1,     "Synonym Type Classification Performance of LLM Systems")
#ontologySubplot(classifiedComplete, "HP:",     outputFileClassEvaluationExpertHPO,    "Synonym Type Classification Performance of LLM Systems (HPO only)")
#ontologySubplot(classifiedComplete, "UBERON:", outputFileClassEvaluationExpertUBERON, "Synonym Type Classification Performance of LLM Systems (UBERON only)")
#ontologySubplot(classifiedComplete, "GO:",     outputFileClassEvaluationExpertGO,     "Synonym Type Classification Performance of LLM Systems (GO only)")
#ontologySubplot(classifiedComplete, "CHEBI:",  outputFileClassEvaluationExpertCHEBI,  "Synonym Type Classification Performance of LLM Systems (CHEBI only)")







combinedEvaluationData = classifiedComplete.copy()

combinedEvaluationData["embeddingLayperson"] = [0] * len(combinedEvaluationData.index)
for laypersonThresholds in embeddingThresholdsLayperson.keys():
    if laypersonThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[laypersonThresholds] <= embeddingThresholdsLayperson[laypersonThresholds], "embeddingLayperson"] += 1
    else:
        log(f"{laypersonThresholds} not found in Columns of Data.")

combinedEvaluationData["embeddingExpert"] = [0] * len(combinedEvaluationData.index)
for expertThresholds in embeddingThresholdsExpert.keys():
    if expertThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[expertThresholds] >= embeddingThresholdsExpert[expertThresholds], "embeddingExpert"] += 1
    else:
        log(f"{expertThresholds} not found in Columns of Data.")






combinedEvaluationAbsoluteData = combinedEvaluationData[combinedEvaluationData[systemColumn] == systems[0]].copy()
combinedEvaluationAbsoluteData["embeddingCorrect"] = (
    ((combinedEvaluationData["embeddingLayperson"] > 0) & (combinedEvaluationData["embeddingExpert"] == 0) & (combinedEvaluationData[typeColumn] == laypersonSynonymType)) |
    ((combinedEvaluationData["embeddingExpert"] > 0) & (combinedEvaluationData["embeddingLayperson"] == 0) & (combinedEvaluationData[typeColumn] == expertSynonymType))
)


discrepanciesCount = len(combinedEvaluationData[(combinedEvaluationData['embeddingLayperson'] > 0) & (combinedEvaluationData['embeddingExpert'] > 0)].index)
log(f"Discrepancies for Embedding Answers: {discrepanciesCount}")

embeddingLaypersonAnswer = combinedEvaluationAbsoluteData[combinedEvaluationAbsoluteData["embeddingLayperson"] > 0]
embeddingExpertAnswer   = combinedEvaluationAbsoluteData[combinedEvaluationAbsoluteData["embeddingExpert"]   > 0]






# Group by votes and correctness, then count occurrences
grouped = embeddingLaypersonAnswer.groupby(["embeddingLayperson", 'embeddingCorrect']).size().unstack(fill_value=0)

# Ensure both columns exist (in case one category is missing)
grouped = grouped.reindex(columns=[False, True], fill_value=0)

# Rename columns for clarity
grouped.columns = ['Wrong', 'Correct']

# Sort by vote count (optional but recommended)
grouped = grouped.sort_index()

# Plot
plt.figure()

plt.bar(grouped.index, grouped['Wrong'], label='Wrong', color='red')
plt.bar(grouped.index, grouped['Correct'],
        bottom=grouped['Wrong'], label='Correct', color='blue')

plt.xlabel('Number of Absolute-"Layperson" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingLaypersonAbsolute, dpi = 300, bbox_inches = "tight")






# Group by votes and correctness, then count occurrences
grouped = embeddingExpertAnswer.groupby(["embeddingExpert", 'embeddingCorrect']).size().unstack(fill_value=0)

# Ensure both columns exist (in case one category is missing)
grouped = grouped.reindex(columns=[False, True], fill_value=0)

# Rename columns for clarity
grouped.columns = ['Wrong', 'Correct']

# Sort by vote count (optional but recommended)
grouped = grouped.sort_index()

# Plot
plt.figure()

plt.bar(grouped.index, grouped['Wrong'], label='Wrong', color='red')
plt.bar(grouped.index, grouped['Correct'],
        bottom=grouped['Wrong'], label='Correct', color='blue')

plt.xlabel('Number of Absolute-"Expert" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingExpertAbsolute, dpi = 300, bbox_inches = "tight")






combinedEvaluationRelaxedData = combinedEvaluationData[combinedEvaluationData[systemColumn] == systems[0]].copy()
combinedEvaluationRelaxedData["embeddingCorrect"] = (
    ((combinedEvaluationData["embeddingLayperson"] > 0) & (combinedEvaluationData[typeColumn] == laypersonSynonymType)) |
    ((combinedEvaluationData["embeddingExpert"] > 0)   & (combinedEvaluationData[typeColumn] == expertSynonymType))
)

embeddingLaypersonAnswer = combinedEvaluationRelaxedData[combinedEvaluationRelaxedData["embeddingLayperson"] > 0]
embeddingExpertAnswer   = combinedEvaluationRelaxedData[combinedEvaluationRelaxedData["embeddingExpert"]   > 0]






# Group by votes and correctness, then count occurrences
grouped = embeddingLaypersonAnswer.groupby(["embeddingLayperson", 'embeddingCorrect']).size().unstack(fill_value=0)

# Ensure both columns exist (in case one category is missing)
grouped = grouped.reindex(columns=[False, True], fill_value=0)

# Rename columns for clarity
grouped.columns = ['Wrong', 'Correct']

# Sort by vote count (optional but recommended)
grouped = grouped.sort_index()

# Plot
plt.figure()

plt.bar(grouped.index, grouped['Wrong'], label='Wrong', color='red')
plt.bar(grouped.index, grouped['Correct'],
        bottom=grouped['Wrong'], label='Correct', color='blue')

plt.xlabel('Number of Relaxed-"Layperson" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingLaypersonRelaxed, dpi = 300, bbox_inches = "tight")






# Group by votes and correctness, then count occurrences
grouped = embeddingExpertAnswer.groupby(["embeddingExpert", 'embeddingCorrect']).size().unstack(fill_value=0)

# Ensure both columns exist (in case one category is missing)
grouped = grouped.reindex(columns=[False, True], fill_value=0)

# Rename columns for clarity
grouped.columns = ['Wrong', 'Correct']

# Sort by vote count (optional but recommended)
grouped = grouped.sort_index()

# Plot
plt.figure()

plt.bar(grouped.index, grouped['Wrong'], label='Wrong', color='red')
plt.bar(grouped.index, grouped['Correct'],
        bottom=grouped['Wrong'], label='Correct', color='blue')

plt.xlabel('Number of Relaxed-"Expert" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingExpertRelaxed, dpi = 300, bbox_inches = "tight")






for index, row in combinedEvaluationData.iterrows():
    if row["embeddingLayperson"] > row["embeddingExpert"]:
        combinedEvaluationData.loc[index, answerColumn] = laypersonSynonymType
    elif row["embeddingLayperson"] < row["embeddingExpert"]:
        combinedEvaluationData.loc[index, answerColumn] = expertSynonymType

embeddingData = combinedEvaluationData[(combinedEvaluationData[systemColumn] == systems[0]) & ((combinedEvaluationData["embeddingLayperson"] > 0) | (combinedEvaluationData["embeddingExpert"] > 0))].copy().reset_index(drop=True)

ontologySubplot(combinedEvaluationData, "",        outputFileCombinedEvaluationRelaxed,     "Synonym Type Classification Performance of combined Approach")






embeddingData['correct'] = embeddingData[answerColumn] == embeddingData[typeColumn]

y_true = embeddingData[typeColumn]
y_pred = embeddingData[answerColumn]

accuracy = accuracy_score(y_true, y_pred)



total = len(embeddingData.index)
correct = embeddingData['correct'].sum()
incorrect = total - correct

class_performance = embeddingData.groupby(typeColumn)['correct'].agg(['sum', 'count'])
class_performance['accuracy'] = class_performance['sum'] / class_performance['count']

plt.figure()
bars = plt.bar(['Correct', 'Incorrect'], [correct, incorrect])

# Add text labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom'
    )

plt.title('Overall Synonym Type Classification Performance')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.grid(axis = "both")
plt.savefig(outputFileClassEmbeddingOutcomeCounts, dpi = 300, bbox_inches = "tight")

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0
)

plt.figure()

classes = sorted(y_true.unique())

x = np.arange(len(classes))
width = 0.3

plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1 Score')

plt.xticks(x, classes)
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Per-Class Performance Metrics')
plt.ylim(0, 1)
plt.grid(axis = "both")
plt.legend(loc = "lower left", ncol = 1)

# Add value labels
for i in range(len(classes)):
    plt.text(x[i] - width, precision[i] + 0.02, f"{precision[i]:.2f}", ha='center')
    plt.text(x[i], recall[i] + 0.02, f"{recall[i]:.2f}", ha='center')
    plt.text(x[i] + width, f1[i] + 0.02, f"{f1[i]:.2f}", ha='center')

plt.savefig(outputFileClassEmbeddingPerformance, dpi = 300, bbox_inches = "tight")


combinedEvaluationData = classifiedComplete.copy()

combinedEvaluationData["embeddingLayperson"] = [0.0] * len(combinedEvaluationData.index)
for laypersonThresholds in embeddingThresholdsLayperson.keys():
    if laypersonThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[laypersonThresholds] <= embeddingThresholdsLayperson[laypersonThresholds], "embeddingLayperson"] += (1.0 / len(embeddingThresholdsLayperson.keys()))
    else:
        log(f"{laypersonThresholds} not found in Columns of Data.")

combinedEvaluationData["embeddingExpert"] = [0.0] * len(combinedEvaluationData.index)
for expertThresholds in embeddingThresholdsExpert.keys():
    if expertThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[expertThresholds] >= embeddingThresholdsExpert[expertThresholds], "embeddingExpert"] += (1.0 / len(embeddingThresholdsExpert.keys()))
    else:
        log(f"{expertThresholds} not found in Columns of Data.")

for col, thresh in embeddingThresholdsLayperson.items():
    if col in combinedEvaluationData.columns:
        pct_below = (combinedEvaluationData[col] <= thresh).mean() * 100
        print(f"{col}: threshold={thresh:.3f}, "
              f"min={combinedEvaluationData[col].min():.3f}, "
              f"max={combinedEvaluationData[col].max():.3f}, "
              f"rows below threshold={pct_below:.1f}%")
    else:
        print(f"{col}: NOT FOUND in columns")

unsureCount = 0
for index, row in combinedEvaluationData.iterrows():
    if row[typeColumn] in synonymTypes:
        if row["embeddingLayperson"] > 0 and row["embeddingExpert"] < (1.0 / len(embeddingThresholdsExpert.keys())):
            combinedEvaluationData.loc[index, answerColumn] = laypersonSynonymType
        elif row["embeddingLayperson"] < (1.0 / len(embeddingThresholdsLayperson.keys())) and row["embeddingExpert"] > 0:
            combinedEvaluationData.loc[index, answerColumn] = expertSynonymType
        elif row["embeddingExpert"] > 0 and row[answerColumn] == expertSynonymType:
            combinedEvaluationData.loc[index, answerColumn] = expertSynonymType
        elif row["embeddingLayperson"] > 0 and row[answerColumn] == laypersonSynonymType:
            combinedEvaluationData.loc[index, answerColumn] = laypersonSynonymType
        else:
            unsureCount += 1
    
log(f"Unsure Count: {unsureCount}")
ontologySubplot(combinedEvaluationData, "",        outputFileCombinedEvaluationAbsolute,     "Synonym Type Classification Performance of combined Approach")


combinedEvaluationData = classifiedComplete.copy()

combinedEvaluationData["embeddingLayperson"] = [0] * len(combinedEvaluationData.index)
for laypersonThresholds in embeddingThresholdsLayperson.keys():
    if laypersonThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[laypersonThresholds] <= embeddingThresholdsLayperson[laypersonThresholds], "embeddingLayperson"] += 1
    else:
        log(f"{laypersonThresholds} not found in Columns of Data.")

combinedEvaluationData["embeddingExpert"] = [0] * len(combinedEvaluationData.index)
for expertThresholds in embeddingThresholdsExpert.keys():
    if expertThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[expertThresholds] >= embeddingThresholdsExpert[expertThresholds], "embeddingExpert"] += 1
    else:
        log(f"{expertThresholds} not found in Columns of Data.")

unsureCount = 0
for index, row in combinedEvaluationData.iterrows():
    if row[typeColumn] in synonymTypes:
        if row["embeddingLayperson"] > 0 or row["embeddingExpert"] > 0:
            if row["embeddingLayperson"] > row["embeddingExpert"]:
                combinedEvaluationData.loc[index, answerColumn] = laypersonSynonymType
            elif row["embeddingLayperson"] < row["embeddingExpert"]:
                combinedEvaluationData.loc[index, answerColumn] = expertSynonymType

ontologySubplot(combinedEvaluationData, "",        outputFileCombinedEvaluationAbsolute2,     "Synonym Type Classification Performance of combined Approach")
