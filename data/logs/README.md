# Logs

| File | Written by | Content |
|---|---|---|
| `syntype_<start>.log` | every script, through the `logger` package | One file per run, named after its start (`logger.file_name`); the first line is the command line of the run. The packages log into the same file. Every step is logged with its start, end and duration, e.g. `Step 'Classifying the synonyms (medgemma-4b-it)' completed in 01:23:45.67.` |
| `prompts_<model>_<experiment>.log` | the `model` package (`Model.logPrompts()`) | Every prompt and answer of one model and experiment, one message per line (`llm.prompt_log_file`) |
| `run_metadata.json` | the `model` package | One record per loaded model: package versions, the model's revision, GPUs, the `llm` configuration, the hashes of the prompt files and the statistics of every generation (`llm.metadata_file`); `run_metadata.json.lock` is its lock file |

Besides the steps, the log of a run shows:

- **Transformation:** the synonyms removed for conflicting gold labels, and those listed several times that count with their first class or type, with the combinations.
- **Development and evaluation set:** how many synonyms of each gold type were drawn from how many, how many concepts, relations and labels of parents and children the development set holds, and how many rows and synonyms the evaluation set keeps.
- **Distances:** for the development and the evaluation set, each embedded on its own: mean, standard deviation, minimum and maximum of the cosine distances, before and after the normalization and per gold type, and the strictly standardized mean difference between expert and layperson synonyms. For the development set also the mean and standard deviation to set as `syntype.cosine_distance_mean` and `_std`.
- **Classification:** the experiment with its settings and prompt files, the number of synonyms and concepts per gold type, the prompt of every turn for the first synonym (in the log file only), how many answers contain a valid classification and a rationale, every wrong or undefined classification with its confidence and rationale (in the log file only), and a summary per model: a table of gold against classified types, precision, recall and F1 per type and the mean confidence.

`cleanUp.sh` deletes the `.log` files.
