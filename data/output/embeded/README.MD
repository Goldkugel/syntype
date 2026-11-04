# Distances

Written by the preparation run (`python3 ./src/SynTypeClassification.py`) from `transform.csv`, and joined to the classifications by `SynTypeEval.py`.

| File | Content |
|---|---|
| `embed.csv` | One row per synonym (synonym classes `exact` and `related`) of all HPO concepts |
| `embed.csv.settings.json` | The settings the file was made with; the distances are computed again when they change or `transform.csv` is newer |

| Column | Content |
|---|---|
| `hpoID`, `class`, `content` | The concept, the synonym class and the synonym |
| `distance_<model>_cosine` | Cosine distance (1 minus the cosine similarity) of the synonym to the label of its concept, for every model of `embedder.model_id`, normalised with `syntype.cosine_distance_mean` and `_std` (unchanged with the defaults 0 and 1, then from 0 to 2); empty for a concept without a label |

All concepts are embedded, also for test runs, so that test and full runs share one scale. The embeddings themselves are cached by the `embedder` package (`embedder.cache_folder`).
