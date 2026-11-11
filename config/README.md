# Configuration

[`config.yaml`](config.yaml) holds the settings of syntype and of the packages it uses, one section each. Paths are relative to the repository root, from where the scripts are run.

| Section | Read by | Content |
|---|---|---|
| `syntype` | `src/config.py` (validated by `src/SynTypeConfig.py`) | The experiments and syntype's own settings (below) |
| `adapter` | the `adapter` package (`HPOAdapter`) | Where `hp.owl` is, and where the transformed HPO goes (`output_folder`, `output_file`); the development and evaluation set are named after it in the same folder |
| `logger` | the `logger` package | Log folder and file name (one file per run), format and runtime display |
| `llm` | the `model` package | The models (`model_id`), sampling and vLLM settings, the prompt templates (`prompt_folder`, `prompt_files`) and the answer store and prompt log (`prompt_tmp_*`, `prompt_log_*`) |
| `embedder` | the `embedder` package | The embedding model of the distances (SapBERT) with its pinned revision, pooling, device and cache |

The settings of the package sections are described in the READMEs of the packages. A key a schema does not know is refused, so typos are found when a run starts.

## `syntype`

| Key | Type | Default | Description |
|---|---|---|---|
| `experiments` | list | (required) | The experiments, picked by their index (counted from 0) on the command line; `start.sh` starts one classification job per experiment, `finalize.sh` merges and evaluates each. Every experiment is a mapping of the settings below |
| `development_set.size_per_type` | `int` (> 0) | `500` | Number of expert and of layperson synonyms drawn for the development set; they are left out of the evaluation set |
| `development_set.seed` | `int` | `0` | Seed of the random selection, so the same development set is drawn again |
| `data_folder` | `str` | `"./data"` | Folder of the data |
| `embedding_model_names` | mapping | `{}` | Name of a model of `embedder.model_id` in the distance column names (`distance_<name>_cosine`); a model without a name is named by its ID without the organisation |
| `cosine_distance_mean` | `float` or `null` | `0` | Mean the cosine distances (1 minus the cosine similarity) are normalised with, `(distance - mean) / std`; `null` takes the mean of the synonyms of each file. Set it to the mean the preparation run logs for the development set |
| `cosine_distance_std` | `float` (> 0) or `null` | `1` | Standard deviation the cosine distances are normalised with; `null` takes that of the synonyms of each file; set it to the one logged for the development set. With `0` and `1` the distances stay as they are (0 to 2); normalised, they can be below -1 and above 1 |
| `distance_evaluation_lower_bound`, `distance_evaluation_upper_bound` | `float` | `-4`, `4` | Range of the normalised distances in the evaluation plots and of the thresholds tried; the lower bound must be below the upper one |
| `distance_evaluation_step` | `float` (> 0) | `0.001` | Distance between two thresholds tried |
| `distance_thresholds` | mapping | `{}` | Per embedding model (named as in the distance columns): the threshold of the normalised cosine distance. The evaluation classifies a synonym at or above it as `layperson` and below it as `expert`, and evaluates this as a model of its own, `<model> (distance)`. The best threshold on the development set is logged by `SynTypeEval.py` ("Distance baseline") |
| `bootstrap_samples` | `int` (> 0) | `1000` | Bootstrap samples of the 95% intervals in the evaluation summary (`..._summary.csv`) |
| `force_draw_additional_plots` | `bool` | `False` | Draw the distance plots also for experiments without `test` |

### Settings of an experiment

A setting that is not given is off.

| Key | Description |
|---|---|
| `test` | Classify only the synonyms of the development set (`hpo.development.csv`, see `development_set`); without it, the synonyms of the evaluation set (`hpo.evaluation.csv`: the expert and layperson synonyms of `hpo.csv` without the development synonyms) are classified |
| `few_shot` | Add the few-shot examples (`fewShotExamples` in `src/config.py`) to the prompts |
| `chain_of_thoughts` | Classify in three prompts (`llm.prompt_files` 1 to 3) instead of one (index 0) |
| `definition` | Add the definition of the concept to the prompts |
| `comment` | Add the comment of the concept to the prompts |
| `structure` | Add the labels of the parents and children of the concept to the prompts |

The settings of an experiment name its files, e.g. `type_Test_Def_NoCom_Str_NoCOT_FS_...` (`Def`/`Com`/`Str`: definition, comment and structure in the prompts; `COT`: chain of thoughts; `FS`: few-shot examples; `No...` if off).
