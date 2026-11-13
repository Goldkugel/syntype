"""
Evaluates the merged classifications of an experiment, with the distance
models of syntype.distance_thresholds as models of their own: a summary CSV
(macro-F1, balanced accuracy, kappa, undefined answers, with bootstrap
intervals) against the majority and distance baselines, confusion matrices,
ROC curves, the confidence calibration of the LLMs, the cross-model agreement
(synonyms correct by 0, 1, ... LLMs) with the synonyms no LLM classified
correctly, counts of the gold and classified types, and precision, recall and
F1 per model and type. The
distance threshold is chosen on the development set by macro-F1.

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

# The plots are written with matplotlib, which does not create the folder.
os.makedirs(os.path.dirname(outputFileClassSummary), exist_ok = True)

with l.step("Evaluating the classifications"):
    exitIfFileNotExist(inputFileClassEvaluation)

    classifiedComplete = readTransformed(inputFileClassEvaluation)

    # The distances of the preparation run: of the development set for test
    # runs, of the evaluation set otherwise. They replace distance columns
    # already in the classifications.
    if isFile(inputFileDistances):
        # Both in the format of the transformed HPO, joined by concept and synonym.
        distances = readCSV(inputFileDistances)
        distances = distances[synonymKeys + [
            c for c in distances.columns if c.startswith(distanceColumnPrePrefix)]]
        for data in [classifiedComplete, distances]:
            data[synonymKeys] = data[synonymKeys].astype(str)

        # many_to_one: a synonym listed twice in the distances would
        # otherwise double its classifications without notice.
        classifiedComplete = classifiedComplete.drop(columns = [
            c for c in classifiedComplete.columns
            if c.startswith(distanceColumnPrePrefix)
        ]).merge(distances, on = synonymKeys, how = "left", validate = "many_to_one")
    else:
        l.log(f"No distances found ({quote(os.path.basename(inputFileDistances))}); " \
            "run SynTypeClassification.py without arguments for the distance plots.")

    # Rows without a model (empty in the CSV, read as NaN) are left out.
    classifiedComplete = classifiedComplete[
        classifiedComplete[systemColumn].fillna("").astype(str) != ""].copy()
    systems = sorted(set(classifiedComplete[systemColumn].astype(str)))
    if len(systems) == 0:
        l.error(f"{quote(os.path.basename(inputFileClassEvaluation))} contains no " \
            "classification of any model.")
        sys.exit(1)

    # Every model classified the same synonyms, so they are counted once.
    ontologies = Counter(
        str(s).split(":", 1)[0] for s in
        classifiedComplete[classifiedComplete[systemColumn] == systems[0]][idColumn]
    )
    l.log(f"Synonyms per ontology ({len(ontologies)}):")
    for ontology, count in ontologies.most_common():
        l.log(f"- {quote(ontology)}: {count}")

    l.log(f"Models ({len(systems)}):")
    for system in systems:
        l.log(f"- {system}")
    l.log(f"Classified synonyms: {len(classifiedComplete.index)} " \
        f"(about {round(len(classifiedComplete.index) / len(systems))} per model).")

    # An empty answer (NaN) counts as undefined.
    classifiedComplete[answerColumn] = classifiedComplete[answerColumn].fillna(
        undefinedSynonymType).astype(str).str.lower()

    # ------------------------------------------------------------------------------
    # Only synonyms of synonymClasses with a gold type of synonymTypes are
    # evaluated; other types (abbreviation, uk, plural, ...) describe the form
    # of a synonym.
    # ------------------------------------------------------------------------------

    classifiedComplete = classifiedComplete[classesOf(classifiedComplete).isin(synonymClasses)]

    goldTypes  = typesOf(classifiedComplete[classifiedComplete[systemColumn] == systems[0]])
    excluded   = goldTypes[~goldTypes.isin(synonymTypes)].value_counts()
    l.log(f"Evaluating the synonyms with gold type {applyFormat(synonymTypes)}; " \
        f"left out: {', '.join(f'{count} {t}' for t, count in excluded.items()) or 'none'}.")

    classifiedComplete = classifiedComplete[
        typesOf(classifiedComplete).isin(synonymTypes)
    ].copy().reset_index(drop = True)

    # The few-shot synonyms are given with their answer in the prompts.
    exampleSynonyms = keysOf(classifiedComplete).isin(
        [(example["id"], example["synonym"]) for example in fewShotExamples])
    l.log(f"Leaving out the {len(fewShotExamples)} synonyms of the few-shot examples " \
        f"({exampleSynonyms.sum()} rows).")
    classifiedComplete = classifiedComplete[~exampleSynonyms].copy().reset_index(drop = True)

    # ------------------------------------------------------------------------------
    # The distance classifiers (syntype.distance_thresholds): layperson if the
    # normalised distance is at or above the threshold, expert below; without
    # a distance, undefined. Evaluated as models of their own, without a
    # confidence.
    # ------------------------------------------------------------------------------

    llmSystems = list(systems)
    reference  = classifiedComplete[classifiedComplete[systemColumn] == systems[0]]
    distanceSystems = {}
    for embeddingModel, threshold in distanceThresholds.items():
        column = distanceColumnPrefix.format(embeddingModel, cosineDistance)
        if column not in reference.columns:
            l.log(f"No {cosineDistance} distances of {quote(embeddingModel)}; " \
                "its threshold is not used.")
            continue
        rows = reference.copy()
        rows[systemColumn] = f"{embeddingModel} (distance)"
        rows[answerColumn] = np.select([rows[column].isna(), rows[column] >= threshold],
            [undefinedSynonymType.lower(), laypersonSynonymType], expertSynonymType)
        rows[confidenceColumn] = np.nan
        rows[rationaleColumn]  = ""
        classifiedComplete = pd.concat([classifiedComplete, rows], ignore_index = True)
        distanceSystems[embeddingModel] = rows[systemColumn].iloc[0]
        l.log(f"Model {quote(distanceSystems[embeddingModel])}:")
        l.log(f"  layperson if the normalised {cosineDistance} distance >= {threshold}, " \
            "expert below.")
    systems = sorted(llmSystems + list(distanceSystems.values()))

    colors = plt.cm.tab10(range(len(systems) + 1))
    color_map = {
        system: plt.cm.tab10(i % 10)
        for i, system in enumerate(systems)
    }
    # One colour per source type in every plot.
    typeColors = {expertSynonymType : colors[0], laypersonSynonymType : colors[1]}

    # The gold type in the plots of the distances.
    goldTypeLabel = "gold type"

    def thresholdScores(distance : np.ndarray, gold : np.ndarray, thresholds) -> np.ndarray:
        """
        Macro-F1 and the F1 of expert and layperson of the distance baseline,
        layperson if the distance is at least the threshold, expert otherwise:
        one row (macro, expert, layperson) per threshold, or one triple for a
        single threshold. With the distances of each type sorted, the
        synonyms at or above a threshold are counted by a binary search, so
        all thresholds take one pass instead of one pass each.
        """
        thresholds = np.atleast_1d(np.asarray(thresholds, dtype = float))
        layperson  = np.sort(distance[gold == laypersonSynonymType])
        expert     = np.sort(distance[gold == expertSynonymType])
        # Per threshold: the layperson synonyms classified as layperson (true
        # positives of layperson) and the expert ones classified as layperson
        # (false positives of layperson); the others are classified as expert.
        laypersonAbove = len(layperson) - np.searchsorted(layperson, thresholds, side = "left")
        expertAbove    = len(expert) - np.searchsorted(expert, thresholds, side = "left")
        expertBelow    = len(expert) - expertAbove
        laypersonBelow = len(layperson) - laypersonAbove
        # F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (predicted + actual), 0 if the
        # type is neither predicted nor in the data.
        f1Layperson = np.divide(2 * laypersonAbove, laypersonAbove + expertAbove + len(layperson),
            out = np.zeros(len(thresholds)), where = laypersonAbove + expertAbove + len(layperson) > 0)
        f1Expert    = np.divide(2 * expertBelow, expertBelow + laypersonBelow + len(expert),
            out = np.zeros(len(thresholds)), where = expertBelow + laypersonBelow + len(expert) > 0)
        scores = np.column_stack([(f1Expert + f1Layperson) / 2, f1Expert, f1Layperson])
        return scores[0] if scores.shape[0] == 1 else scores

    # ------------------------------------------------------------------------------
    # Distance baseline: one threshold for both source types, chosen on the
    # development set by macro-F1 (each type counts the same, whatever its
    # share), and the plots of the distances (test runs or
    # force_draw_additional_plots). Test runs evaluate on the development set
    # itself, the others apply its threshold to the evaluation set (tuned on
    # it if embed.development.csv is missing).
    # ------------------------------------------------------------------------------

    gold = classifiedComplete[classifiedComplete[systemColumn] == llmSystems[0]].copy().reset_index(drop = True)
    development = gold
    if not reduceToTestIDs and isFile(outputFileEmbeddedDevelopment):
        development = readTransformed(outputFileEmbeddedDevelopment)
        development = development[typesOf(development).isin(synonymTypes)]
    drawPlots = reduceToTestIDs or forceDrawAdditionalPlots
    # The predictions of the distance baselines (embedding models without a
    # configured threshold) and the distances, per gold synonym, for the
    # summary and the ROC curves.
    baselines      = {}
    distanceScores = {}

    for embeddingModel in embeddingModels.keys():
        for distanceMetric in distanceMetrics:
            column = distanceColumnPrefix.format(embeddingModel, distanceMetric)
            if column not in gold.columns or column not in development.columns:
                continue

            # The distances with the gold type of every synonym.
            tuning     = pd.DataFrame({column : development[column],
                goldTypeLabel : typesOf(development)}).dropna()
            evaluation = pd.DataFrame({column : gold[column],
                goldTypeLabel : typesOf(gold)}).dropna()
            if len(tuning.index) == 0 or len(evaluation.index) == 0:
                continue

            # Every threshold of the configured range, widened to the data.
            thresholds = np.linspace(
                min(distanceEvaluationLowerBound, tuning[column].min()),
                max(distanceEvaluationUpperBound, tuning[column].max()),
                distanceEvaluationParts)
            scores = thresholdScores(tuning[column].to_numpy(),
                tuning[goldTypeLabel].to_numpy(), thresholds)
            best = thresholds[np.argmax(scores[:, 0])]
            # A configured threshold is evaluated as a model instead.
            if embeddingModel not in distanceSystems:
                baselines[f"distance ({embeddingModel})"] = np.where(
                    gold[column].to_numpy() >= best, laypersonSynonymType, expertSynonymType)
            distanceScores[f"distance ({embeddingModel})"] = gold[column].to_numpy()

            macro, expert, layperson = thresholdScores(evaluation[column].to_numpy(),
                evaluation[goldTypeLabel].to_numpy(), best)
            share = 100 * np.mean(evaluation[column].to_numpy() >= best)
            l.log(f"Distance baseline ({embeddingModel}):")
            l.log(f"  layperson if the normalised {distanceMetric} distance >= {best:.3f} " \
                "(chosen on the development set).")
            l.log(f"  macro-F1 {macro:.3f} on the " \
                f"{'development' if reduceToTestIDs else 'evaluation'} set.")
            l.log(f"  F1 expert {expert:.3f}, layperson {layperson:.3f}; " \
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
            sns.kdeplot(data = evaluation, x = column, hue = goldTypeLabel, palette = typeColors,
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
    # and the distance threshold): macro-F1, balanced accuracy, Cohen's kappa,
    # accuracy and the share of undefined answers, with 95% bootstrap intervals.
    # Undefined answers count as wrong.
    # ------------------------------------------------------------------------------


    def confusionCounts(goldTypes : np.ndarray, predicted : np.ndarray, samples : int = 0) -> np.ndarray:
        """
        The confusion counts gold (expert, layperson) x answer (expert,
        layperson, other), of the data and of `samples` bootstrap samples of it
        (shape (1 + samples, 2, 3)); seeded, so the intervals are reproducible.
        Drawing n synonyms with replacement only changes how often each of
        the 6 cells is drawn, so a bootstrap sample is drawn directly as
        multinomial counts of the cells (the same distribution, without
        resampling every synonym).
        """
        codes  = 3 * (goldTypes == laypersonSynonymType) + np.select(
            [predicted == expertSynonymType, predicted == laypersonSynonymType], [0, 1], 2)
        counts = np.bincount(codes, minlength = 6)
        rng    = np.random.default_rng(0)
        drawn  = rng.multinomial(counts.sum(), counts / counts.sum(), size = samples)
        return np.vstack([counts, drawn]).reshape(-1, 2, 3)

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

    goldTypes = typesOf(gold).to_numpy()
    majority  = typesOf(gold).value_counts().idxmax()
    goldIndex = pd.MultiIndex.from_frame(gold[synonymKeys])
    predictions = {
        system : classifiedComplete[classifiedComplete[systemColumn] == system].set_index(
            synonymKeys)[answerColumn].reindex(goldIndex).to_numpy()
        for system in systems
    }
    predictions[f"majority ({majority})"] = np.full(len(goldTypes), majority)
    predictions.update(baselines)

    summarising = f"Summarising {len(systems)} models and {len(predictions) - len(systems)} baselines"
    l.log(f"{summarising}...")
    l.log(f"{len(goldTypes)} synonyms; intervals from {bootstrapSamples} bootstrap samples.")
    rows = []
    for system, predicted in predictions.items():
        metrics = summaryMetrics(confusionCounts(goldTypes, predicted, bootstrapSamples))
        row = {systemColumn : system, "synonyms" : len(goldTypes)}
        for name, values in metrics.items():
            row[name] = values[0]
            row[f"{name}_low"], row[f"{name}_high"] = np.percentile(values[1:], [2.5, 97.5])
        rows.append(row)
        interval = lambda name: f"{row[name]:.3f} [{row[name + '_low']:.3f}, {row[name + '_high']:.3f}]"
        l.log(f"{system}:")
        l.log(f"  macro-F1 {interval('macro_f1')}, undefined {100 * row['undefined']:.1f}%.")
        l.log(f"  balanced accuracy {interval('balanced_accuracy')}, kappa {interval('kappa')}.")
    writeHugeCSV(pd.DataFrame(rows), outputFileClassSummary)
    l.log(f"{summarising} completed.")

    # ------------------------------------------------------------------------------
    # Cross-model agreement: per gold type, how many synonyms 0, 1, ... of the
    # LLMs (not the baselines) classified correctly; an undefined answer
    # counts as wrong. The synonyms no LLM classified correctly are listed as
    # candidates for a review by the curators.
    # ------------------------------------------------------------------------------

    agreeing = f"Counting the agreement of the {len(llmSystems)} LLMs"
    l.log(f"{agreeing}...")
    correctCount = np.sum([predictions[system] == goldTypes for system in llmSystems], axis = 0)
    modelCounts  = range(len(llmSystems) + 1)
    agreement = []
    for goldType in synonymTypes + ["total"]:
        selected = np.ones(len(goldTypes), dtype = bool) if goldType == "total" else goldTypes == goldType
        counts   = np.bincount(correctCount[selected], minlength = len(modelCounts))
        for count in modelCounts:
            agreement.append({goldTypeLabel : goldType, "models correct" : count,
                "synonyms" : counts[count], "share" : counts[count] / max(selected.sum(), 1)})
        l.log(f"{goldType} ({selected.sum()} synonyms), correct by 0-{len(llmSystems)} models: " + \
            ", ".join(f"{counts[count]} ({100 * counts[count] / max(selected.sum(), 1):.0f}%)"
            for count in modelCounts) + ".")
    writeHugeCSV(pd.DataFrame(agreement), outputFileClassAgreement)

    # The label of every concept, from the transformed HPO, if there is one.
    labels = labelsByConcept(readCSV(outputFileTransformedFull)) if isFile(outputFileTransformedFull) else {}
    misclassified = gold.loc[correctCount == 0, synonymKeys].copy()
    misclassified.insert(1, labelClass, misclassified[idColumn].map(labels).fillna(""))
    misclassified[goldTypeLabel] = goldTypes[correctCount == 0]
    for system in llmSystems:
        misclassified[system] = predictions[system][correctCount == 0]
    writeHugeCSV(misclassified.sort_values([goldTypeLabel, idColumn]), outputFileClassMisclassified)
    l.log(f"{len(misclassified.index)} synonyms no LLM classified correctly: " + ", ".join(
        f"{count} {goldType}" for goldType, count in misclassified[goldTypeLabel].value_counts().items()) + \
        f"; written to {quote(os.path.basename(outputFileClassMisclassified))}.")
    l.log(f"{agreeing} completed.")

    # Confusion matrices of the models: gold type x answer (expert, layperson),
    # as counts and as shares of all synonyms of the gold type; undefined
    # answers are left out (their share is in the summary).
    columnsCount = min(3, len(systems))
    fig, axes = plt.subplots(nrows = math.ceil(len(systems) / columnsCount), ncols = columnsCount,
        figsize = (4.5 * columnsCount, 3.5 * math.ceil(len(systems) / columnsCount)), squeeze = False)
    for ax, system in zip(axes.flat, systems):
        counts = confusionCounts(goldTypes, predictions[system])[0]
        shares = counts / np.maximum(counts.sum(axis = 1, keepdims = True), 1)
        counts, shares = counts[:, :2], shares[:, :2]
        sns.heatmap(shares, ax = ax, vmin = 0, vmax = 1, cmap = "Blues", cbar = False,
            annot = np.array([[f"{c}\n({100 * r:.0f}%)" for c, r in zip(cr, rr)] for cr, rr in zip(counts, shares)]),
            fmt = "", xticklabels = synonymTypes, yticklabels = synonymTypes)
        ax.set_title(system, fontsize = 9)
        ax.set_xlabel("Answer")
        ax.set_ylabel("Gold type")
    for ax in list(axes.flat)[len(systems):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices (share of all synonyms of the gold type)")
    plt.tight_layout()
    plt.savefig(outputFileClassConfusion, dpi = 300, bbox_inches = "tight")
    plt.close()

    # ROC curves with layperson as the positive type: a model ranks a synonym
    # by its confidence, positive for layperson answers, negative for expert
    # ones (0 if undefined); the distances by their value (the median if
    # missing).
    isLayperson = goldTypes == laypersonSynonymType
    if 0 < isLayperson.sum() < len(isLayperson):
        rankings = {}
        for system in llmSystems:
            data = classifiedComplete[classifiedComplete[systemColumn] == system].set_index(
                synonymKeys).reindex(goldIndex)
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
    # confidence level (1 to 5) of every LLM; the distance models have no
    # confidence.
    levels = np.array(confidenceLevels)
    fig, (top, bottom) = plt.subplots(nrows = 2, figsize = (7, 7), sharex = True)
    for index, system in enumerate(llmSystems):
        data = classifiedComplete[classifiedComplete[systemColumn] == system]
        confidence = pd.to_numeric(data[confidenceColumn], errors = "coerce")
        correct    = data[answerColumn] == typesOf(data)
        accuracy   = [correct[confidence == level].mean() if (confidence == level).any() else np.nan
            for level in levels]
        share      = [(confidence == level).mean() for level in levels]
        top.plot(levels, accuracy, marker = "o", color = color_map[system], label = system)
        bottom.bar(levels + (index - (len(systems) - 1) / 2) * 0.8 / len(systems), share,
            width = 0.8 / len(systems), color = color_map[system])
        l.log(f"{system}: accuracy per confidence {levels[0]}-{levels[-1]}: " + ", ".join(
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

    types = Counter(typesOf(gold))

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
    # Classification counts per system, including undefined answers.
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

    def performancePlot(data : pd.DataFrame, outputFile : str, title : str) -> None:
        """
        Plots precision, recall and F1 per model and source type, and logs
        them.
        """
        classified = data.reset_index(drop = True)
        result = {}

        systems = sorted(set(data[systemColumn].tolist()))
        for system in systems:
            systemData = classified[classified[systemColumn] == system]

            # An undefined answer lowers the recall of its gold type and no
            # precision; its share is logged. A type that is never predicted
            # (or not in the gold data) scores 0.
            precision, recall, f1, _ = precision_recall_fscore_support(
                typesOf(systemData), systemData[answerColumn],
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
            l.log(f"{system}: {100 * undefined:.1f}% of {len(systemData.index)} answers undefined.")
            for t in synonymTypes:
                l.log(f"{system}, {quote(t)}: " \
                    f"precision {result[system][t][precisionLabel]:.3f}, " \
                    f"recall {result[system][t][recallLabel]:.3f}, " \
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

    performancePlot(classifiedComplete, outputFileClassRecallPrecisionF1,
        "Source Type Classification Performance of the Models")
