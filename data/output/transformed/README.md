# Transformed HPO

Written by the preparation run (`python3 ./src/SynTypeClassification.py`): `transform.csv` from `hp.owl` with the `adapter` package, and the development set `hpo.reduced.csv` from `transform.csv` (also by `python3 ./src/SynTypeDevelopmentSet.py`). Read by the embedding and the classification.

| File | Content |
|---|---|
| `transform.csv` | All concepts of the HPO |
| `hpo.reduced.csv` | The development set, classified by runs with `test`: `syntype.development_set.size_per_type` expert and layperson synonyms each (500 by default), drawn at random with `syntype.development_set.seed`, with the label, definition and comment of their concepts, the child rows of these concepts (to their children and from their parents) and the labels of those parents and children. No other synonym is in it |
| `<file>.settings.json` | The settings the file was made with; the HPO is transformed again when they change or `hp.owl` is newer, and the development set is drawn again when its settings change or `transform.csv` is newer |

Both CSV files are in the format of the `adapter` package: one row per `id`, `attribute` and `value`, with a dict of additional information in `additional`. The attributes are `label`, `definition`, `comment`, `child` (the value is the ID of a child concept), `reference` and `synonym`. A synonym's `additional` holds its synonym class and source type, e.g. `{'semantic_class': 'exact', 'source_type': 'layperson'}`; a synonym listed several times for a concept has all of them, joined with `", "`, and counts with the first one (`synonymClassOf()` and `synonymTypeOf()` in `src/SynTypeUtils.py`).

Compared with the output of the adapter, synonyms listed both as exact and related synonym, or both as layperson synonym and abbreviation, and synonyms equal to the label of their concept are removed. `readTransformed()` in `src/SynTypeUtils.py` reads a file with the `additional` dicts.
