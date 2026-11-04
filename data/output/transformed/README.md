# Transformed HPO

Written by the preparation run (`python3 ./src/SynTypeClassification.py`) from `hp.owl` with the `adapter` package, and read by the embedding and the classification.

| File | Content |
|---|---|
| `transform.csv` | All concepts of the HPO |
| `transform.reduced.csv` | Only the test concepts (`testIDs` in `src/config.py`), their parents and their children, used by runs with `test` |
| `<file>.settings.json` | The settings the file was made with; the HPO is transformed again when they change or `hp.owl` is newer |

Both CSV files are in the format of the `adapter` package: one row per `id`, `attribute` and `value`, with a dict of additional information in `additional`. The attributes are `label`, `definition`, `comment`, `child` (the value is the ID of a child concept), `reference` and `synonym`. A synonym's `additional` holds its synonym class and source type, e.g. `{'semantic_class': 'exact', 'source_type': 'layperson'}`; a synonym listed several times for a concept has all of them, joined with `", "`, and counts with the first one (`synonymClassOf()` and `synonymTypeOf()` in `src/SynTypeUtils.py`).

Compared with the output of the adapter, synonyms listed both as exact and related synonym, or both as layperson synonym and abbreviation, and synonyms equal to the label of their concept are removed. `readTransformed()` in `src/SynTypeUtils.py` reads a file with the `additional` dicts.
