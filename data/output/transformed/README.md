# Transformed HPO

Written by the preparation run (`python3 ./src/SynTypeClassification.py`): `hpo.csv` from `hp.owl` with the `adapter` package, and from it the development set `hpo.development.csv` and the evaluation set `hpo.evaluation.csv` (also by `python3 ./src/SynTypeDevelopmentSet.py`). Read by the embedding and the classification.

| File | Content |
|---|---|
| `hpo.csv` | All concepts of the HPO |
| `hpo.development.csv` | The development set, classified by runs with `test`: `syntype.development_set.size_per_type` expert and layperson synonyms each (500 by default), drawn at random with `syntype.development_set.seed`, with the label, definition and comment of their concepts, the child rows of these concepts (to their children and from their parents) and the labels of those parents and children. No other synonym is in it |
| `hpo.evaluation.csv` | The evaluation set, classified by runs without `test`: every row of `hpo.csv` except those of the synonyms of the development set (a synonym listed several times is removed with all its rows), so that no development synonym is evaluated, and except the synonyms of other gold types than `expert` and `layperson` (e.g. abbreviations, plurals; counted with their first type) |
| `<file>.settings.json` | The settings the file was made with; the HPO is transformed again when they change or `hp.owl` is newer, and the development and evaluation set are drawn again when the settings of the development set change or `hpo.csv` is newer |

The CSV files are in the format of the `adapter` package: one row per `id`, `attribute` and `value`, with a dict of additional information in `additional`. The attributes are `label`, `definition`, `comment`, `child` (the value is the ID of a child concept), `reference` and `synonym`. A synonym's `additional` holds its synonym class and source type, e.g. `{'semantic_class': 'exact', 'source_type': 'layperson'}`; a synonym listed several times for a concept has all of them, joined with `", "`, and counts with the first one (`synonymClassOf()` and `synonymTypeOf()` in `src/SynTypeUtils.py`).

Compared with the output of the adapter, synonyms listed both as exact and related synonym, or both as layperson synonym and abbreviation, and synonyms equal to the label of their concept are removed. `readTransformed()` in `src/SynTypeUtils.py` reads a file with the `additional` dicts.
