# Evaluation

Written by `python3 ./src/SynTypeEval.py <index>` from the merged classifications of an experiment (`type_<experiment>_...`). Only synonyms whose gold source type is `expert` or `layperson` are evaluated, and the synonyms of the few-shot examples are left out, also in runs without examples.

| File | Content |
|---|---|
| `..._gold_counts.png` | Number of synonyms per gold source type |
| `..._answer_counts.png` | Number of classifications per source type and model |
| `..._base_evaluation.png` | Precision, recall and F1 per model and source type, over all concepts |
| `..._exact_HPO_evaluation.png` | The same for HPO concepts only (IDs `HP:`); `UBERON`, `GO` and `CHEBI` the same for concepts of those ontologies, if the data has any |
| `type_<Test>_distance_evaluation_distance_<model>_cosine.png` | Density of the normalised distances (see [`../embeded/README.MD`](../embeded/README.MD)) per gold source type |
| `type_<Test>_embedding_evaluation_distance_<model>_cosine.png` | The distance baseline on the development set: macro-F1 and the F1 per type of classifying by one threshold (layperson if the distance is at least the threshold, i.e. further from the label; expert otherwise), per threshold within `syntype.distance_evaluation_lower_bound` and `_upper_bound` (widened to the data), with the best threshold marked |

The distance baseline chooses its threshold on the development set by macro-F1, so that both types count the same whatever their share. Runs with `test` are scored on the development set itself (the threshold is chosen on the same synonyms); the other runs apply the threshold to the evaluation set (`embed.development.csv` gives it). The threshold and the scores are logged for every run; the distance plots are drawn for experiments with `test`, or for all with `syntype.force_draw_additional_plots`, and drawn again on every run.

The logs also give, per model and ontology, the share of undefined answers: an undefined answer lowers the recall of its gold type but no precision.
