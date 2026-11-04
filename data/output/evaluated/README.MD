# Evaluation

Written by `python3 ./src/SynTypeEval.py <index>` from the merged classifications of an experiment (`type_<experiment>_...`). Only synonyms whose gold source type is `expert` or `layperson` are evaluated, and the synonyms of the few-shot examples are left out, also in runs without examples.

| File | Content |
|---|---|
| `..._gold_counts.png` | Number of synonyms per gold source type |
| `..._answer_counts.png` | Number of classifications per source type and model |
| `..._base_evaluation.png` | Precision, recall and F1 per model and source type, over all concepts |
| `..._exact_HPO_evaluation.png` | The same for HPO concepts only (IDs `HP:`); `UBERON`, `GO` and `CHEBI` the same for concepts of those ontologies, if the data has any |
| `type_<Test>_distance_evaluation_distance_<model>_cosine.png` | Density of the normalised distances (see [`../embeded/README.MD`](../embeded/README.MD)) per gold source type |
| `type_<Test>_embedding_evaluation_distance_<model>_cosine.png` | F1 of classifying by a distance threshold (layperson above, i.e. further from the label; expert below), per threshold within `syntype.distance_evaluation_lower_bound` and `_upper_bound`; the best thresholds are logged |

The distance plots are drawn for experiments with `test`, or for all with `syntype.force_draw_additional_plots`; they are not drawn again if they exist.
