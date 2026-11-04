# Raw classifications

Written by the classification (`python3 ./src/SynTypeClassification.py <index>`): one file per experiment and model, `type_<experiment>_raw_<model>.csv`, written after every turn of the conversation. `<experiment>` names the settings of the experiment, e.g. `Test_Definition_NoComment_Structure_NoChainOfThoughts_FewShot`.

| Column | Content |
|---|---|
| `hpoID`, `class`, `type`, `content` | The concept, the synonym class (`exact` or `related`), the gold source type from the HPO (`expert`, `layperson`, `abbreviation`, `plural`, `uk`, ...) and the synonym |
| `system` | The model |
| `user1`, `answer` | The prompt and the answer (single prompt) |
| `user1`, `assistant1`, `user2`, `assistant2`, `user3`, `answer` | The prompts and answers of the three turns (chain-of-thoughts) |
| `classification`, `confidence` | The classification (`expert`, `layperson`, or `undefined` for an answer without a valid one) and its confidence (1 to 5, `-1` if undefined) parsed from the last answer |

The prompts and answers themselves are stored by the `model` package in its answer store (`data/tmp/`), from which an interrupted run resumes.
