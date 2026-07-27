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
from model      import *
from utils      import *
from config     import *

from collections import Counter

printHeader("Evaluating the Results of Synonym Classification")
startTime = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassEvaluation)

classifiedComplete = readCSV(inputFileClassEvaluation)







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
                        exactSynonymClass : colors[0],
                        relatedSynonymClass : colors[1]
                    }

                # Draw density curves for each class
                sns.kdeplot(
                    data=classified,
                    x=column,
                    palette=palette,
                    hue=classified[classColumn],
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
                            classified[classColumn] == relatedSynonymClass
                        ))
                        FP = sum((
                            classified[column] <= i
                                ) & (
                            classified[classColumn] != relatedSynonymClass
                        ))

                        if TP + FP > 0:
                            y1.append((2 * TP) / (2 * TP + FP))
                        else:
                            y1.append(np.nan)

                        TP = sum((
                            classified[column] >= i
                                ) & (
                            classified[classColumn] == exactSynonymClass
                        ))
                        FP = sum((
                            classified[column] >= i
                                ) & (
                            classified[classColumn] != exactSynonymClass
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
                plt.plot(x2, y2, label = "F1 Score (Exact Threshold)",   
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
                    log(f"Suggested Threshold for Class {quote(exactSynonymClass)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)} with F1 of {max_y} classifying {str(p)}% of the data." )

                plt.xlabel("Threshold")
                plt.ylabel("F1 Score")
                plt.title(f"Normalized {similarityMetric.capitalize()}-Similarity for Model {quote(embeddingModel)}")
                plt.grid(axis = "both")

                plt.yticks(np.linspace(0, 1.0, 11))
                plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 17))

                plt.legend(loc = "lower left", ncol = 1)
                plt.savefig(outputFileClassEmbeddingEval, dpi = 300, bbox_inches = "tight")






# Plot Gold Counts
classified = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy().drop([answerColumn, systemColumn], axis = 1).drop_duplicates().reset_index(drop = True)
classes = Counter(classified[classColumn])

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

plt.xlabel("Semantic Class")
plt.ylabel("Count")
plt.title("Count of Semantic Classes")
plt.grid(axis = "both")
plt.show()
plt.savefig(outputFileClassGoldCounts, dpi = 300, bbox_inches = "tight")






# Plot Classification Counts
classifiedComplete = classifiedComplete[classifiedComplete[classColumn].isin(synonymClasses)].copy().reset_index(drop = True)

# Count occurrences per system and classification
classified = classifiedComplete
counts = classifiedComplete.groupby([answerColumn, systemColumn]).size().unstack(fill_value = 0)
# Create plot
ax = counts.plot(kind = "bar", figsize = (3 * len(systems), 4), width = 0.8)

# Add value labels above bars
for container in ax.containers:
    ax.bar_label(container, padding = 3)

plt.xlabel("Classified Semantic Class")
plt.ylabel("Count")
plt.title("Count of Classified Semantic Class")
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
            for synonymClasse in synonymClasses:
                
                systemClassResults = {
                    precisionLabel  : 0,
                    recallLabel     : 0,
                    f1ScoreLabel    : 0
                }

                if len(systemData[systemData[answerColumn] == 
                    synonymClasse].index) > 0:

                    systemClassResults[precisionLabel] = \
                        len(systemData[
                                (systemData[classColumn] == synonymClasse
                                    ) & (
                                systemData[answerColumn] == synonymClasse)
                            ].index
                        ) / (
                            len(systemData[systemData[answerColumn] == 
                                synonymClasse].index) 
                        )
                    
                    systemClassResults[recallLabel] = \
                        len(systemData[
                                (systemData[classColumn] == synonymClasse
                                    ) & (
                                systemData[answerColumn] == synonymClasse)
                            ].index
                        ) / (
                            len(systemData[systemData[classColumn] == 
                                synonymClasse].index) 
                        )
                        
                    if systemClassResults[recallLabel] > 0 or systemClassResults[precisionLabel] > 0:
                        systemClassResults[f1ScoreLabel] = \
                            2 * systemClassResults[precisionLabel] * \
                            systemClassResults[recallLabel] / (
                            systemClassResults[recallLabel] + 
                            systemClassResults[precisionLabel])
                    else:
                        systemClassResults[f1ScoreLabel] = 0
                else:
                    systemClassResults[f1ScoreLabel]    = 0
                    systemClassResults[recallLabel]     = 0
                    systemClassResults[precisionLabel]  = 0

                systemResults[synonymClasse] = systemClassResults

        result[system] = systemResults

    metrics = [f1ScoreLabel, recallLabel, precisionLabel]
    systems = list(result.keys())
    classes = list(next(iter(result.values())).keys())

    x = np.arange(len(systems)) * 0.1 * len(systems)
    barWidth = 0.2

    fig, axes = plt.subplots(
        nrows = len(metrics),
        ncols = len(classes),
        figsize = (3 * len(metrics), 2 * len(classes)),
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
ontologySubplot(classifiedComplete, "",        outputFileClassRecallPrecisionF1,     "Semantic Class Classification Performance of LLM Systems")
ontologySubplot(classifiedComplete, "HP:",     outputFileClassEvaluationExactHPO,    "Semantic Class Classification Performance of LLM Systems (HPO only)")
ontologySubplot(classifiedComplete, "UBERON:", outputFileClassEvaluationExactUBERON, "Semantic Class Classification Performance of LLM Systems (UBERON only)")
ontologySubplot(classifiedComplete, "GO:",     outputFileClassEvaluationExactGO,     "Semantic Class Classification Performance of LLM Systems (GO only)")
ontologySubplot(classifiedComplete, "CHEBI:",  outputFileClassEvaluationExactCHEBI,  "Semantic Class Classification Performance of LLM Systems (CHEBI only)")







combinedEvaluationData = classifiedComplete.copy()

combinedEvaluationData["embeddingRelated"] = [0] * len(combinedEvaluationData.index)
for relatedThresholds in embeddingThresholdsRelated.keys():
    if relatedThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[relatedThresholds] <= embeddingThresholdsRelated[relatedThresholds], "embeddingRelated"] += 1
    else:
        log(f"{relatedThresholds} not found in Columns of Data.")

combinedEvaluationData["embeddingExact"] = [0] * len(combinedEvaluationData.index)
for exactThresholds in embeddingThresholdsExact.keys():
    if exactThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[exactThresholds] >= embeddingThresholdsExact[exactThresholds], "embeddingExact"] += 1
    else:
        log(f"{relatedThresholds} not found in Columns of Data.")






combinedEvaluationAbsoluteData = combinedEvaluationData[combinedEvaluationData[systemColumn] == systems[0]].copy()
combinedEvaluationAbsoluteData["embeddingCorrect"] = (
    ((combinedEvaluationData["embeddingRelated"] > 0) & (combinedEvaluationData["embeddingExact"] == 0) & (combinedEvaluationData[classColumn] == relatedSynonymClass)) |
    ((combinedEvaluationData["embeddingExact"] > 0) & (combinedEvaluationData["embeddingRelated"] == 0) & (combinedEvaluationData[classColumn] == exactSynonymClass))
)


discrepanciesCount = len(combinedEvaluationData[(combinedEvaluationData['embeddingRelated'] > 0) & (combinedEvaluationData['embeddingExact'] > 0)].index)
log(f"Discrepancies for Embedding Answers: {discrepanciesCount}")

embeddingRelatedAnswer = combinedEvaluationAbsoluteData[combinedEvaluationAbsoluteData["embeddingRelated"] > 0]
embeddingExactAnswer   = combinedEvaluationAbsoluteData[combinedEvaluationAbsoluteData["embeddingExact"]   > 0]






# Group by votes and correctness, then count occurrences
grouped = embeddingRelatedAnswer.groupby(["embeddingRelated", 'embeddingCorrect']).size().unstack(fill_value=0)

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

plt.xlabel('Number of Absolute-"Related" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingRelatedAbsolute, dpi = 300, bbox_inches = "tight")






# Group by votes and correctness, then count occurrences
grouped = embeddingExactAnswer.groupby(["embeddingExact", 'embeddingCorrect']).size().unstack(fill_value=0)

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

plt.xlabel('Number of Absolute-"Exact" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingExactAbsolute, dpi = 300, bbox_inches = "tight")






combinedEvaluationRelaxedData = combinedEvaluationData[combinedEvaluationData[systemColumn] == systems[0]].copy()
combinedEvaluationRelaxedData["embeddingCorrect"] = (
    ((combinedEvaluationData["embeddingRelated"] > 0) & (combinedEvaluationData[classColumn] == relatedSynonymClass)) |
    ((combinedEvaluationData["embeddingExact"] > 0)   & (combinedEvaluationData[classColumn] == exactSynonymClass))
)

embeddingRelatedAnswer = combinedEvaluationRelaxedData[combinedEvaluationRelaxedData["embeddingRelated"] > 0]
embeddingExactAnswer   = combinedEvaluationRelaxedData[combinedEvaluationRelaxedData["embeddingExact"]   > 0]






# Group by votes and correctness, then count occurrences
grouped = embeddingRelatedAnswer.groupby(["embeddingRelated", 'embeddingCorrect']).size().unstack(fill_value=0)

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

plt.xlabel('Number of Relaxed-"Related" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingRelatedRelaxed, dpi = 300, bbox_inches = "tight")






# Group by votes and correctness, then count occurrences
grouped = embeddingExactAnswer.groupby(["embeddingExact", 'embeddingCorrect']).size().unstack(fill_value=0)

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

plt.xlabel('Number of Relaxed-"Exact" Votes')
plt.ylabel('Frequency')
plt.title('Vote Distribution (Correct vs Wrong)')
plt.grid(axis = "both")
plt.legend()

plt.savefig(outputFileClassEmbeddingExactRelaxed, dpi = 300, bbox_inches = "tight")






for index, row in combinedEvaluationData.iterrows():
    if row["embeddingRelated"] > row["embeddingExact"]:
        combinedEvaluationData.loc[index, answerColumn] = relatedSynonymClass
    elif row["embeddingRelated"] < row["embeddingExact"]:
        combinedEvaluationData.loc[index, answerColumn] = exactSynonymClass

embeddingData = combinedEvaluationData[(combinedEvaluationData[systemColumn] == systems[0]) & ((combinedEvaluationData["embeddingRelated"] > 0) | (combinedEvaluationData["embeddingExact"] > 0))].copy().reset_index(drop=True)

ontologySubplot(combinedEvaluationData, "",        outputFileCombinedEvaluationRelaxed,     "Semantic Class Classification Performance of combined Approach")






embeddingData['correct'] = embeddingData[answerColumn] == embeddingData[classColumn]

y_true = embeddingData[classColumn]
y_pred = embeddingData[answerColumn]

accuracy = accuracy_score(y_true, y_pred)



total = len(embeddingData.index)
correct = embeddingData['correct'].sum()
incorrect = total - correct

class_performance = embeddingData.groupby(classColumn)['correct'].agg(['sum', 'count'])
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

plt.title('Overall Semantic Class Classification Performance')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.grid(axis = "both")
plt.savefig(outputFileClassEmbeddingOutcomeCounts, dpi = 300, bbox_inches = "tight")

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
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

combinedEvaluationData["embeddingRelated"] = [0.0] * len(combinedEvaluationData.index)
for relatedThresholds in embeddingThresholdsRelated.keys():
    if relatedThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[relatedThresholds] <= embeddingThresholdsRelated[relatedThresholds], "embeddingRelated"] += (1.0 / len(embeddingThresholdsRelated.keys()))
    else:
        log(f"{relatedThresholds} not found in Columns of Data.")

combinedEvaluationData["embeddingExact"] = [0.0] * len(combinedEvaluationData.index)
for exactThresholds in embeddingThresholdsExact.keys():
    if exactThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[exactThresholds] >= embeddingThresholdsExact[exactThresholds], "embeddingExact"] += (1.0 / len(embeddingThresholdsExact.keys()))
    else:
        log(f"{relatedThresholds} not found in Columns of Data.")

unsureCount = 0
for index, row in combinedEvaluationData.iterrows():
    if row[classColumn] in synonymClasses:
        if row["embeddingRelated"] > 0 and row["embeddingExact"] < (1.0 / len(embeddingThresholdsExact.keys())):
            combinedEvaluationData.loc[index, answerColumn] = relatedSynonymClass
        elif row["embeddingRelated"] < (1.0 / len(embeddingThresholdsRelated.keys())) and row["embeddingExact"] > 0:
            combinedEvaluationData.loc[index, answerColumn] = exactSynonymClass
        elif row["embeddingExact"] > 0 and row[answerColumn] == exactSynonymClass:
            combinedEvaluationData.loc[index, answerColumn] = exactSynonymClass
        elif row["embeddingRelated"] > 0 and row[answerColumn] == relatedSynonymClass:
            combinedEvaluationData.loc[index, answerColumn] = relatedSynonymClass
        else:
            unsureCount += 1
    
log(f"Unsure Count: {unsureCount}")
ontologySubplot(combinedEvaluationData, "",        outputFileCombinedEvaluationAbsolute,     "Semantic Class Classification Performance of combined Approach")


combinedEvaluationData = classifiedComplete.copy()

combinedEvaluationData["embeddingRelated"] = [0] * len(combinedEvaluationData.index)
for relatedThresholds in embeddingThresholdsRelated.keys():
    if relatedThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[relatedThresholds] <= embeddingThresholdsRelated[relatedThresholds], "embeddingRelated"] += 1
    else:
        log(f"{relatedThresholds} not found in Columns of Data.")

combinedEvaluationData["embeddingExact"] = [0] * len(combinedEvaluationData.index)
for exactThresholds in embeddingThresholdsExact.keys():
    if exactThresholds in combinedEvaluationData.columns:
        combinedEvaluationData.loc[combinedEvaluationData[exactThresholds] >= embeddingThresholdsExact[exactThresholds], "embeddingExact"] += 1
    else:
        log(f"{relatedThresholds} not found in Columns of Data.")

unsureCount = 0
for index, row in combinedEvaluationData.iterrows():
    if row[classColumn] in synonymClasses:
        if row["embeddingRelated"] > 0 or row["embeddingExact"] > 0:
            if row["embeddingRelated"] > row["embeddingExact"]:
                combinedEvaluationData.loc[index, answerColumn] = relatedSynonymClass
            elif row["embeddingRelated"] < row["embeddingExact"]:
                combinedEvaluationData.loc[index, answerColumn] = exactSynonymClass

ontologySubplot(combinedEvaluationData, "",        outputFileCombinedEvaluationAbsolute2,     "Semantic Class Classification Performance of combined Approach")
