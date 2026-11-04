# Raw classifications

Written by the classification (`python3 ./src/SynTypeClassification.py <index>`): one file per experiment and model, `type_<experiment>_raw_<model>.csv`, written after every turn of the conversation. `<experiment>` names the settings of the experiment, e.g. `Test_Def_NoCom_Str_NoCOT_FS`.

| Column | Content |
|---|---|
| `id`, `attribute`, `value`, `additional` | The synonym's row of the transformed HPO (`hpo.development.csv` or `hpo.evaluation.csv`): the concept, `synonym`, the synonym and its additional information, which holds the synonym class (`semantic_class`: `exact` or `related`) and the gold source type (`source_type`: `expert`, `layperson`, or `direct`, which counts as `expert`) |
| `system` | The model |
| `user1`, `answer` | The prompt and the answer (single prompt) |
| `user1`, `assistant1`, `user2`, `assistant2`, `user3`, `answer` | The prompts and answers of the three turns (chain-of-thoughts) |
| `classification`, `confidence` | The classification (`expert`, `layperson`, or `undefined` for an answer without a valid one) and its confidence (1 to 5, `-1` if undefined) parsed from the last answer |
| `rationale` | The short explanation of the decision in the last answer (`""` if it has none) |

The prompts and answers themselves are stored by the `model` package in its answer store (`data/tmp/`), from which an interrupted run resumes.
