# Logs

| File | Written by | Content |
|---|---|---|
| `syntype_<start>.log` | every script, through the `logger` package | One file per run, named after its start (`logger.file_name`); the first line is the command line of the run. The packages log into the same file. Every step is logged with its start, end and duration, e.g. `Step 'Classifying the synonyms (medgemma-4b-it)' completed in 01:23:45.67.` |
| `prompts_<model>_<experiment>.log` | the `model` package (`Model.logPrompts()`) | Every prompt and answer of one model and experiment, one message per line (`llm.prompt_log_file`) |
| `run_metadata.json` | the `model` package | One record per loaded model: package versions, the model's revision, GPUs, the `llm` configuration, the hashes of the prompt files and the statistics of every generation (`llm.metadata_file`); `run_metadata.json.lock` is its lock file |

`cleanUp.sh` deletes the `.log` files.
