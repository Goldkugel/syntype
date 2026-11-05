# Evaluation

Written by `python3 ./src/SynTypeEval.py <index>` from the merged classifications of an experiment (`type_<experiment>_...`). Only synonyms whose gold source type is `expert` or `layperson` are evaluated, and the synonyms of the few-shot examples are left out, also in runs without examples.

| File | Content |
|---|---|
| `..._gold_counts.png` | Number of synonyms per gold source type |
| `..._answer_counts.png` | Number of classifications per source type and model |
| `..._base_evaluation.png` | Precision, recall and F1 per model and source type |
| `..._summary.csv` | One row per model and baseline (`majority (<type>)`: always the most frequent gold type; `distance (<model>)`: the distance threshold of the development set): macro-F1, balanced accuracy, Cohen's kappa, accuracy and the share of undefined answers, each with the bounds of its 95% interval from 1000 bootstrap samples (`_low`, `_high`); also logged |
| `..._confusion_matrices.png` | Gold type × answer (expert, layperson) per model, as counts and as shares of all synonyms of the gold type; undefined answers are left out (their share is in `..._summary.csv`) |
| `..._roc.png` | ROC curves and AUC with layperson as the positive type: a model ranks the synonyms by its confidence (positive for layperson answers, negative for expert ones), the baseline by the distance; the AUCs are logged |
| `..._calibration.png` | Accuracy and share of the answers per confidence level (1 to 5) of every model; the accuracies are logged |
| `type_<Test>_distance_evaluation_distance_<model>_cosine.png` | Density of the normalised distances (see [`../embedded/README.md`](../embedded/README.md)) per gold source type |
| `type_<Test>_embedding_evaluation_distance_<model>_cosine.png` | The distance baseline on the development set: macro-F1 and the F1 per type of classifying by one threshold (layperson if the distance is at least the threshold, i.e. further from the label; expert otherwise), per threshold within `syntype.distance_evaluation_lower_bound` and `_upper_bound` (widened to the data), with the best threshold marked |

With a threshold in `syntype.distance_thresholds`, the distance classifier `<model> (distance)` (layperson at or above the threshold, expert below) is evaluated like the LLMs: in the summary, the confusion matrices, the answer counts and the precision, recall and F1 plot; having no confidence, it is not in the calibration plot, and the ROC curves rank by the distance itself. It then replaces the `distance (<model>)` baseline in the summary.

The distance baseline chooses its threshold on the development set by macro-F1, so that both types count the same whatever their share. Runs with `test` are scored on the development set itself (the threshold is chosen on the same synonyms); the other runs apply the threshold to the evaluation set (`embed.development.csv` gives it). The threshold and the scores are logged for every run; the distance plots are drawn for experiments with `test`, or for all with `syntype.force_draw_additional_plots`, and drawn again on every run.

The logs also give, per model and ontology, the share of undefined answers: an undefined answer lowers the recall of its gold type but no precision.
