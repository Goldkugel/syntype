# Logs

Written by every script through the `logger` package (`logger` in [`config/config.yaml`](../../config/config.yaml)) and by the `model` package (`llm.prompt_log_*`, `llm.metadata_*`).

| File | Content |
|---|---|
| `syntype_<start>.log` | The log of one run, named after its start to the microsecond: the command line as its first row, then every step with its start, end and duration. The classification lists the wrong and undefined classifications of every model with their confidence and rationale, followed by a summary of the model (gold against classified types, precision, recall and F1); the evaluation logs the scores of the models and baselines |
| `prompts_<model>_<experiment>.log` | Every prompt and answer of a model in an experiment (`model.logPrompts()`) |
| `run_metadata.json` | The settings of the runs of the `model` package: model revisions, sampling settings and the hashes of the prompt templates |

`cleanUp.sh` deletes the `.log` files.
