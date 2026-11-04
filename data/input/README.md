# Input

The HPO release to classify, read by the preparation run (`python3 ./src/SynTypeClassification.py`) with the `adapter` package (`adapter.hpo` in [`config/config.yaml`](../../config/config.yaml)):

| File | Content |
|---|---|
| `hp.owl` | The Human Phenotype Ontology in OWL, e.g. from the [HPO releases](https://github.com/obophenotype/human-phenotype-ontology/releases) |

The file is not tracked. When it is replaced by a newer one, the next preparation run transforms the HPO and computes the distances again.
