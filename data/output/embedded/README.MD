# Distances

Written by the preparation run (`python3 ./src/SynTypeClassification.py`), and joined to the classifications by `SynTypeEval.py`: `embed.development.csv` to those of runs with `test`, `embed.csv` to the others.

| File | Content |
|---|---|
| `embed.development.csv` | One row per synonym of the development set (`hpo.development.csv`), embedded on its own |
| `embed.csv` | One row per synonym (synonym classes `exact` and `related`) of the evaluation set (`hpo.evaluation.csv`) |
| `<model>-<hash>.json` | The embedding cache of the `embedder` package (`embedder.cache_folder`): the vector of every term encoded so far, per model and settings (the hash); kept by `cleanUp.sh` |
| `<file>.settings.json` | The settings the file was made with; the distances are computed again when they change or its input is newer |

| Column | Content |
|---|---|
| `id`, `attribute`, `value`, `additional` | The synonym's row of the transformed HPO, as in `hpo.development.csv` and `hpo.evaluation.csv`: the concept, `synonym`, the synonym and its additional information (synonym class and source type) |
| `distance_<model>_cosine` | Cosine distance (1 minus the cosine similarity) of the synonym to the label of its concept, for every model of `embedder.model_id`, normalised with `syntype.cosine_distance_mean` and `_std` (unchanged with the defaults 0 and 1, then from 0 to 2); empty for a concept without a label |

The development set is embedded first: the preparation run logs the mean and standard deviation of its distances, which are to be set as `syntype.cosine_distance_mean` and `_std`, so that the distances of the evaluation set are normalised with them. Changing them computes both files again. The embeddings themselves are cached in this folder by the `embedder` package (`embedder.cache_folder`), so a term is encoded only once.
