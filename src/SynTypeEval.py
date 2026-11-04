"""
Evaluates the merged classifications of an experiment: counts of the gold and
classified source types, precision, recall and F1 per model and source type,
and the cosine distance of the synonyms to their label per source type.

    python3 ./src/SynTypeEval.py <index>   (index in syntype.experiments)
"""

import numpy                as np
import matplotlib.pyplot    as plt
import seaborn              as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from matplotlib.lines import Line2D
import sys
import shlex
import math

sys.dont_write_bytecode = True

from SynTypeUtils import *
from config     import *

from logger     import Logger

from collections import Counter

# The log file is named after the start of the run only, so the command line
# is its first line.
l             = Logger(config = configFile)
l.log(f"Command: python3 {shlex.join(sys.argv)}")

if experimentIndex is None:
    l.error("Give the index of the experiment in syntype.experiments, e.g. " \
        f"python3 {sys.argv[0]} 0.")
    sys.exit(1)

l.printHeader("Evaluating the Results of Synonym Classification")

with l.step("Evaluating the classifications"):
    exitIfFileNotExist(inputFileClassEvaluation)

    classifiedComplete = readCSV(inputFileClassEvaluation)

    # The distances of the preparation run: of the development set for test
    # runs, of the evaluation set otherwise. They replace distance columns
    # already in the classifications.
    if isFile(inputFileDistances):
        # In the format of the transformed HPO; joined by concept, synonym class
        # and synonym.
        distances = readTransformed(inputFileDistances)
        distances[hpoidColumn]   = distances[idColumn]
        distances[classColumn]   = distances[additionalColumn].map(synonymClassOf)
        distances[contentColumn] = distances[valueColumn]
        keys         = [hpoidColumn, classColumn, contentColumn]
        distances    = distances[keys + [
            c for c in distances.columns if c.startswith(distanceColumnPrePrefix)]]
        for data in [classifiedComplete, distances]:
            data[keys] = data[keys].astype(str)

        classifiedComplete = classifiedComplete.drop(columns = [
            c for c in classifiedComplete.columns
            if c.startswith(distanceColumnPrePrefix)
        ]).merge(distances, on = keys, how = "left")
    else:
        l.log(f"No distances found ({os.path.basename(inputFileDistances)}); " \
            "run SynTypeClassification.py without arguments for the distance plots.")

    systems = list(set(classifiedComplete[systemColumn].tolist()))
    systems = sorted(systems)

    # Every model classified the same synonyms, so they are counted once.
    ontologies = [
        str(s).split(":", 1)[0] for s in
        classifiedComplete[classifiedComplete[systemColumn] == systems[0]][hpoidColumn].to_list()
    ]

    counts = Counter(ontologies)

    l.log(f"{len(counts.keys())} ontologies found.")

    for key in counts.keys():
        l.log(f"Found {counts[key]} synonyms of concepts of the ontology '{key}'.")

    l.log(f"Found {len(systems)} systems:")
    for system in systems:
        l.log(f"- {system}")
    l.log(f"Classified Synonyms: {len(classifiedComplete.index)} " \
        f"(~{int(len(classifiedComplete.index) / len(systems))} per system)")
    classifiedComplete = classifiedComplete[classifiedComplete[systemColumn] != ""]

    classifiedComplete[classColumn]   = classifiedComplete[classColumn ].str.lower()
    classifiedComplete[answerColumn]  = classifiedComplete[answerColumn].str.lower()
    classifiedComplete[typeColumn]    = classifiedComplete[typeColumn  ].str.lower()

    # ------------------------------------------------------------------------------
    # Only synonyms with a gold type of synonymTypes are evaluated; other types
    # (abbreviation, uk, plural, ...) describe the form of a synonym.
    # ------------------------------------------------------------------------------

    classifiedComplete = classifiedComplete[classifiedComplete[classColumn].isin(synonymClasses)]

    goldSystem = classifiedComplete[classifiedComplete[systemColumn] == systems[0]]
    excluded   = goldSystem[~goldSystem[typeColumn].isin(synonymTypes)][typeColumn].value_counts()
    l.log(f"Evaluating the synonyms with gold type {applyFormat(synonymTypes)}; " \
        f"left out: {', '.join(f'{count} {t}' for t, count in excluded.items()) or 'none'}.")

    classifiedComplete = classifiedComplete[
        classifiedComplete[typeColumn].isin(synonymTypes)
    ].copy().reset_index(drop = True)

    # The Few-Shot synonyms are given with their answer in the prompts.
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
    # One colour per source type in every plot.
    typeColors = {expertSynonymType : colors[0], laypersonSynonymType : colors[1]}

    def typeF1(predicted : np.ndarray, gold : np.ndarray, sourceType : str) -> float:
        """F1 of sourceType (0 if it is neither predicted nor in the gold data)."""
        tp = np.sum((predicted == sourceType) & (gold == sourceType))
        fp = np.sum((predicted == sourceType) & (gold != sourceType))
        fn = np.sum((predicted != sourceType) & (gold == sourceType))
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0

    def thresholdScores(distance : np.ndarray, gold : np.ndarray, threshold : float) -> tuple[float, float, float]:
        """
        Macro-F1 and the F1 of expert and layperson of the distance baseline:
        layperson if the distance is at least the threshold, expert otherwise.
        """
        predicted = np.where(distance >= threshold, laypersonSynonymType, expertSynonymType)
        expert    = typeF1(predicted, gold, expertSynonymType)
        layperson = typeF1(predicted, gold, laypersonSynonymType)
        return (expert + layperson) / 2, expert, layperson

    # ------------------------------------------------------------------------------
    # Distance baseline: one threshold for both source types, chosen on the
    # development set by macro-F1 (each type counts the same, whatever its
    # share), and the density of the distances per gold type. Test runs
    # evaluate on the development set itself, the others apply its threshold
    # to the evaluation set.
    # ------------------------------------------------------------------------------

    gold = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy().reset_index(drop = True)
    development = gold
    if not reduceToTestIDs and isFile(outputFileEmbeddedDevelopment):
        development = readTransformed(outputFileEmbeddedDevelopment)
        development[typeColumn] = development[additionalColumn].map(synonymTypeOf)
        development = development[development[typeColumn].isin(synonymTypes)]
    drawPlots = reduceToTestIDs or forceDrawAdditionalPlots
    # The predictions of the distance baselines and the distances, per gold
    # synonym, for the summary and the ROC curves.
    baselines      = {}
    distanceScores = {}

    for embeddingModel in embeddingModels.keys():
        for distanceMetric in distanceMetrics:
            column = distanceColumnPrefix.format(embeddingModel, distanceMetric)
            if column not in gold.columns or column not in development.columns:
                continue

            tuning     = development[[column, typeColumn]].dropna()
            evaluation = gold[[column, typeColumn]].dropna()
            if len(tuning.index) == 0 or len(evaluation.index) == 0:
                continue

            # Every threshold of the configured range, widened to the data.
            thresholds = np.linspace(
                min(distanceEvaluationLowerBound, tuning[column].min()),
                max(distanceEvaluationUpperBound, tuning[column].max()),
                distanceEvaluationParts)
            scores = np.array([thresholdScores(tuning[column].to_numpy(),
                tuning[typeColumn].to_numpy(), t) for t in thresholds])
            best = thresholds[np.argmax(scores[:, 0])]
            baselines[f"distance ({embeddingModel})"] = np.where(
                gold[column].to_numpy() >= best, laypersonSynonymType, expertSynonymType)
            distanceScores[f"distance ({embeddingModel})"] = gold[column].to_numpy()

            macro, expert, layperson = thresholdScores(evaluation[column].to_numpy(),
                evaluation[typeColumn].to_numpy(), best)
            share = 100 * np.mean(evaluation[column].to_numpy() >= best)
            l.log(f"Distance baseline ({embeddingModel}): layperson if the normalised " \
                f"{distanceMetric} distance >= {best:.3f}.")
            l.log(f"Threshold chosen on the development set; macro-F1 {macro:.3f} on the " \
                f"{'development' if reduceToTestIDs else 'evaluation'} set.")
            l.log(f"F1 expert {expert:.3f}, layperson {layperson:.3f}; " \
                f"{share:.1f}% classified as layperson.")

            if not drawPlots:
                continue

            outputFileClassEmbeddingEval = os.path.join(dataDir, outputFolderName,
                outputFolderNameClassEmbedding, outputFileNameClassEmbeddingEvaluation.format(column))
            outputFileClassDistanceEval  = os.path.join(dataDir, outputFolderName,
                outputFolderNameClassEmbedding, outputFileNameClassDistanceEvaluation.format(column))

            plotting = f"Plotting the {distanceMetric} distance of {embeddingModel}"
            l.log(f"{plotting}...")

            plt.figure(figsize = (8, 5))
            sns.kdeplot(data = evaluation, x = column, hue = typeColumn, palette = typeColors,
                common_norm = False, fill = False)
            plt.xlabel("Normalised distance")
            plt.ylabel("Density")
            plt.title("Density of the Distance by Source Type")
            plt.xticks(np.linspace(distanceEvaluationLowerBound, distanceEvaluationUpperBound, 9))
            plt.xlim(distanceEvaluationLowerBound, distanceEvaluationUpperBound)
            plt.legend(handles = [Line2D([0], [0], color = color, lw = 2, label = t)
                for t, color in typeColors.items()], title = "Source Type")
            plt.grid(axis = "both")
            plt.tight_layout()
            plt.savefig(outputFileClassDistanceEval, dpi = 300, bbox_inches = "tight")
            plt.close()

            # The scores of every threshold on the development set.
            plt.figure()
            plt.plot(thresholds, scores[:, 0], label = "Macro-F1", color = "black")
            plt.plot(thresholds, scores[:, 1], label = "F1 expert", color = typeColors[expertSynonymType])
            plt.plot(thresholds, scores[:, 2], label = "F1 layperson", color = typeColors[laypersonSynonymType])
            plt.axvline(best, color = "grey", linestyle = "--",
                label = f"Threshold {best:.3f} (macro-F1 {scores[:, 0].max():.3f})")
            plt.xlabel("Threshold (layperson if the distance is at least the threshold)")
            plt.ylabel("F1 score on the development set")
            plt.title(f"Normalised {distanceMetric.capitalize()} Distance of {quote(embeddingModel)}")
            plt.grid(axis = "both")
            plt.yticks(np.linspace(0, 1.0, 11))
            plt.xlim(distanceEvaluationLowerBound, distanceEvaluationUpperBound)
            plt.legend(loc = "lower left", ncol = 1)
            plt.savefig(outputFileClassEmbeddingEval, dpi = 300, bbox_inches = "tight")
            plt.close()
            l.log(f"{plotting} completed.")

    # ------------------------------------------------------------------------------
    # Summary of every model and of the baselines (the most frequent gold type,
    # and the distance threshold): macro-F1, balanced accuracy, Cohen's kappa
    # and the share of undefined answers, with 95% bootstrap intervals.
    # ------------------------------------------------------------------------------

    answers = synonymTypes + [undefinedSynonymType.lower()]

    def confusionCounts(goldTypes : np.ndarray, predicted : np.ndarray, samples : int = 0) -> np.ndarray:
        """
        The confusion counts gold (expert, layperson) x answer (expert,
        layperson, other), of the data and of `samples` bootstrap samples of it
        (shape (1 + samples, 2, 3)).
        """
        codes = 3 * (goldTypes == laypersonSynonymType) + np.select(
            [predicted == expertSynonymType, predicted == laypersonSynonymType], [0, 1], 2)
        rng = np.random.default_rng(0)
        counts = [np.bincount(codes, minlength = 6)]
        for _ in range(samples):
            counts.append(np.bincount(codes[rng.integers(0, len(codes), len(codes))], minlength = 6))
        return np.array(counts).reshape(-1, 2, 3)

    def summaryMetrics(counts : np.ndarray) -> dict:
        """The metrics of confusion counts (see confusionCounts()), per sample."""
        n         = counts.sum(axis = (1, 2))
        actual    = counts.sum(axis = 2)
        columns   = counts.sum(axis = 1)
        tp        = counts[:, [0, 1], [0, 1]]
        f1        = 2 * tp / np.maximum(actual + columns[:, :2], 1)
        accuracy  = tp.sum(axis = 1) / n
        # Chance agreement for kappa; an undefined answer never agrees.
        expected  = (actual * columns[:, :2]).sum(axis = 1) / n ** 2
        return {
            "macro_f1"          : f1.mean(axis = 1),
            "balanced_accuracy" : (tp / np.maximum(actual, 1)).mean(axis = 1),
            "kappa"             : (accuracy - expected) / np.where(expected < 1, 1 - expected, 1),
            "accuracy"          : accuracy,
            "undefined"         : columns[:, 2] / n,
        }

    goldTypes = gold[typeColumn].to_numpy()
    majority  = gold[typeColumn].value_counts().idxmax()
    predictions = {
        system : classifiedComplete[classifiedComplete[systemColumn] == system].set_index(
            [hpoidColumn, classColumn, contentColumn])[answerColumn].reindex(
            pd.MultiIndex.from_frame(gold[[hpoidColumn, classColumn, contentColumn]])).to_numpy()
        for system in systems
    }
    predictions[f"majority ({majority})"] = np.full(len(goldTypes), majority)
    predictions.update(baselines)

    summarising = f"Summarising {len(systems)} models and {len(predictions) - len(systems)} baselines"
    l.log(f"{summarising}...")
    l.log(f"{len(goldTypes)} synonyms; intervals from 1000 bootstrap samples.")
    rows = []
    for system, predicted in predictions.items():
        metrics = summaryMetrics(confusionCounts(goldTypes, predicted, 1000))
        row = {systemColumn : system, "synonyms" : len(goldTypes)}
        for name, values in metrics.items():
            row[name] = values[0]
            row[f"{name}_low"], row[f"{name}_high"] = np.percentile(values[1:], [2.5, 97.5])
        rows.append(row)
        interval = lambda name: f"{row[name]:.3f} [{row[name + '_low']:.3f}, {row[name + '_high']:.3f}]"
        l.log(f"{system}: macro-F1 {interval('macro_f1')}, undefined {100 * row['undefined']:.1f}%.")
        l.log(f"{system}: balanced accuracy {interval('balanced_accuracy')}, kappa {interval('kappa')}.")
    writeHugeCSV(pd.DataFrame(rows), outputFileClassSummary)
    l.log(f"{summarising} completed.")

    # Confusion matrices of the models, as counts and as shares of the gold type.
    columnsCount = min(3, len(systems))
    fig, axes = plt.subplots(nrows = math.ceil(len(systems) / columnsCount), ncols = columnsCount,
        figsize = (4.5 * columnsCount, 3.5 * math.ceil(len(systems) / columnsCount)), squeeze = False)
    for ax, system in zip(axes.flat, systems):
        counts = confusionCounts(goldTypes, predictions[system])[0]
        shares = counts / np.maximum(counts.sum(axis = 1, keepdims = True), 1)
        sns.heatmap(shares, ax = ax, vmin = 0, vmax = 1, cmap = "Blues", cbar = False,
            annot = np.array([[f"{c}\n({100 * r:.0f}%)" for c, r in zip(cr, rr)] for cr, rr in zip(counts, shares)]),
            fmt = "", xticklabels = answers, yticklabels = synonymTypes)
        ax.set_title(system, fontsize = 9)
        ax.set_xlabel("Answer")
        ax.set_ylabel("Gold type")
    for ax in list(axes.flat)[len(systems):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices (share of the gold type)")
    plt.tight_layout()
    plt.savefig(outputFileClassConfusion, dpi = 300, bbox_inches = "tight")
    plt.close()

    # ROC curves with layperson as the positive type: a model ranks a synonym
    # by its confidence, positive for layperson answers, negative for expert
    # ones (0 if undefined); the baseline by the distance.
    isLayperson = goldTypes == laypersonSynonymType
    if 0 < isLayperson.sum() < len(isLayperson):
        rankings = {}
        for system in systems:
            data = classifiedComplete[classifiedComplete[systemColumn] == system].set_index(
                [hpoidColumn, classColumn, contentColumn]).reindex(
                pd.MultiIndex.from_frame(gold[[hpoidColumn, classColumn, contentColumn]]))
            confidence = pd.to_numeric(data[confidenceColumn], errors = "coerce").fillna(0).clip(lower = 0)
            rankings[system] = np.select([data[answerColumn] == laypersonSynonymType,
                data[answerColumn] == expertSynonymType], [confidence, -confidence], 0)
        rankings.update({name : np.nan_to_num(values, nan = np.nanmedian(values))
            for name, values in distanceScores.items()})

        plt.figure(figsize = (6, 6))
        for name, scores in rankings.items():
            auc = roc_auc_score(isLayperson, scores)
            falsePositives, truePositives, _ = roc_curve(isLayperson, scores)
            plt.plot(falsePositives, truePositives, label = f"{name} (AUC {auc:.3f})",
                color = color_map.get(name, "black"), linestyle = "-" if name in systems else "--")
            l.log(f"{name}: ROC AUC {auc:.3f} (layperson as positive).")
        plt.plot([0, 1], [0, 1], color = "grey", linestyle = ":", label = "Chance")
        plt.xlabel("False positive rate (expert synonyms classified as layperson)")
        plt.ylabel("True positive rate (layperson synonyms found)")
        plt.title("ROC Curves, Layperson as Positive")
        plt.legend(loc = "lower right", fontsize = 7)
        plt.grid(axis = "both")
        plt.savefig(outputFileClassROC, dpi = 300, bbox_inches = "tight")
        plt.close()

    # Confidence calibration: the accuracy and the share of the answers per
    # confidence level (1 to 5) of every model.
    levels = np.arange(1, 6)
    fig, (top, bottom) = plt.subplots(nrows = 2, figsize = (7, 7), sharex = True)
    for index, system in enumerate(systems):
        data = classifiedComplete[classifiedComplete[systemColumn] == system]
        confidence = pd.to_numeric(data[confidenceColumn], errors = "coerce")
        correct    = data[answerColumn] == data[typeColumn]
        accuracy   = [correct[confidence == level].mean() if (confidence == level).any() else np.nan
            for level in levels]
        share      = [(confidence == level).mean() for level in levels]
        top.plot(levels, accuracy, marker = "o", color = color_map[system], label = system)
        bottom.bar(levels + (index - (len(systems) - 1) / 2) * 0.8 / len(systems), share,
            width = 0.8 / len(systems), color = color_map[system])
        l.log(f"{system}: accuracy per confidence 1-5: " + ", ".join(
            "-" if np.isnan(a) else f"{a:.2f}" for a in accuracy) + ".")
    top.set_ylabel("Accuracy")
    top.set_ylim(0, 1)
    top.grid(axis = "both")
    top.legend(fontsize = 7)
    top.set_title("Accuracy and Share of the Answers per Confidence")
    bottom.set_ylabel("Share of the answers")
    bottom.set_xlabel("Confidence")
    bottom.set_xticks(levels)
    bottom.grid(axis = "both")
    plt.tight_layout()
    plt.savefig(outputFileClassCalibration, dpi = 300, bbox_inches = "tight")
    plt.close()

    # ------------------------------------------------------------------------------
    # Gold counts.
    # ------------------------------------------------------------------------------

    classified = classifiedComplete[classifiedComplete[systemColumn] == systems[0]].copy().drop([answerColumn, systemColumn], axis = 1).drop_duplicates().reset_index(drop = True)
    types = Counter(classified[typeColumn])

    labels = [t for t in synonymTypes if t in types]
    values = [types[t] for t in labels]

    plt.figure()
    bars = plt.bar(labels, values, color = [typeColors[t] for t in labels])

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
    plt.close()

    # ------------------------------------------------------------------------------
    # Classification counts per system.
    # ------------------------------------------------------------------------------

    counts = classifiedComplete.groupby([answerColumn, systemColumn]).size().unstack(fill_value = 0)
    ax = counts.plot(kind = "bar", figsize = (3 * len(systems), 4), width = 0.8)

    for container in ax.containers:
        ax.bar_label(container, padding = 3)

    plt.xlabel("Classified Source Type")
    plt.ylabel("Count")
    plt.title("Count of Classified Source Types")
    plt.xticks(rotation = 0)
    plt.grid(axis = "both")
    plt.savefig(outputFileClassAnswerCounts, dpi = 300, bbox_inches = "tight")
    plt.close()

    # ------------------------------------------------------------------------------
    # Precision, recall and F1 per system and source type.
    # ------------------------------------------------------------------------------

    def ontologySubplot(data : pd.DataFrame = None, startString : str = "", outputFile : str = "", title : str = "") -> None:
        """
        Plots precision, recall and F1 per model and source type for the
        synonyms of the concepts whose ID starts with `startString` ("" for
        all), and logs them; does nothing if there are none.
        """
        classified = data[data[hpoidColumn].str.startswith(startString, na=False)].reset_index(drop = True).copy()

        # E.g. the development set may have only HPO concepts.
        if len(classified.index) == 0:
            l.log(f"No synonyms of concepts starting with {quote(startString)}; skipping the plot {quote(title)}.")
            return

        result = {}

        systems = sorted(set(data[systemColumn].tolist()))
        for system in systems:
            systemData = classified[classified[systemColumn] == system]

            # An undefined answer lowers the recall of its gold type and no
            # precision; its share is logged. A type that is never predicted
            # (or not in the gold data) scores 0.
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

            undefined = (systemData[answerColumn] == undefinedSynonymType.lower()).mean()
            l.log(f"{system} ({quote(startString or 'all')}): {100 * undefined:.1f}% " \
                f"of {len(systemData.index)} answers undefined.")
            for t in synonymTypes:
                l.log(f"{system} ({quote(startString or 'all')}), {t}: " \
                    f"P {result[system][t][precisionLabel]:.3f}, " \
                    f"R {result[system][t][recallLabel]:.3f}, " \
                    f"F1 {result[system][t][f1ScoreLabel]:.3f}.")

        metrics = [f1ScoreLabel, recallLabel, precisionLabel]
        classes = list(synonymTypes)

        x = np.arange(len(systems)) * 0.1 * len(systems)
        barWidth = 0.2

        fig, axes = plt.subplots(
            nrows = len(metrics),
            ncols = len(classes),
            figsize = (3 * len(classes), 2 * len(metrics)),
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
        plt.close()

    # All synonyms, then per ontology of the concept.
    ontologySubplot(classifiedComplete, "",        outputFileClassRecallPrecisionF1,     "Source Type Classification Performance of LLM Systems")
    ontologySubplot(classifiedComplete, "HP:",     outputFileClassEvaluationExactHPO,    "Source Type Classification Performance of LLM Systems (HPO only)")
    ontologySubplot(classifiedComplete, "UBERON:", outputFileClassEvaluationExactUBERON, "Source Type Classification Performance of LLM Systems (UBERON only)")
    ontologySubplot(classifiedComplete, "GO:",     outputFileClassEvaluationExactGO,     "Source Type Classification Performance of LLM Systems (GO only)")
    ontologySubplot(classifiedComplete, "CHEBI:",  outputFileClassEvaluationExactCHEBI,  "Source Type Classification Performance of LLM Systems (CHEBI only)")
