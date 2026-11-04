# Source

The scripts of syntype, run from the repository root (see the [README](../README.md)).

| File | Content |
|---|---|
| `SynTypeClassification.py` | Without arguments the preparation run: transforms `hp.owl` with the `adapter` package into `transform.csv` and `transform.reduced.csv` and embeds the labels and synonyms with the `embedder` package into `embed.csv`, each only if needed. With an experiment index: classifies the synonyms with every model of `llm.model_id` (or the one given) through the `model` package and formats the answers |
| `SynTypeMerge.py` | Merges the formatted classifications of all models of an experiment into one file |
| `SynTypeEval.py` | Evaluates the merged classifications of an experiment: counts, precision, recall and F1 per model and source type, and distance plots |
| `SynTypeUtils.py` | Helpers shared by the scripts: reading and writing CSV files, the transformed HPO in the adapter's format (`readTransformed()`, `synonymTable()`, `synonymClassOf()`, `synonymTypeOf()`), parsing the answers (`parseClassification()`), progress bars and the data summary |
| `SynTypeConfig.py` | Pydantic schema of the `syntype` section of `config/config.yaml` and of its experiments |
| `config.py` | Reads the configuration, picks the experiment and the models from the command line, defines the column and class names, the file names of every step, the few-shot examples and the test concepts. `python3 ./src/config.py` prints the number of experiments |

The scripts import `config.py` and `SynTypeUtils.py` with `from ... import *`. The packages of a step (`adapter`, `embedder`, `model`) are imported by the step itself, so that the preparation run does not load vLLM and the classification does not load the embedding models.

Each script starts with the logger: the first line of its log file is its command line, and every step is logged with its start, end and duration (`Logger.step()`).
