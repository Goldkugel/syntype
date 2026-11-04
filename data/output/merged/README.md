# Merged classifications

Written by `python3 ./src/SynTypeMerge.py <index>`: the formatted classifications (see [`../formatted/README.md`](../formatted/README.md)) of the models of `llm.model_id` for an experiment in one file, `type_<experiment>_merged_classes.csv`, one row per synonym and model (`system`). Read by `SynTypeEval.py`.

Only the models of `llm.model_id` are merged: a model without a formatted classification (e.g. a job that failed or is still running) is logged as a warning, and files of models no longer in the list are left out, also with a warning. If the models classified different synonyms (e.g. of another HPO version), the differences are logged. Without any classification the merge stops with an error.
