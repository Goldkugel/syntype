# Formatted classifications

Written by the classification (`python3 ./src/SynTypeClassification.py <index>`) after a model has classified all synonyms: one file per experiment and model, `type_<experiment>_formatted_<model>.csv`, read by `SynTypeMerge.py`.

The columns are those of the raw classification (see [`../type/README.md`](../type/README.md)), except that `answer` holds the classification (`expert`, `layperson` or `undefined`) and `confidence` its confidence (`-1` if undefined); the `classification` column is left out. Wrong and undefined classifications are listed in the log with their confidence and rationale, followed by a summary of the model (gold against classified types, precision, recall and F1), see [`../../logs/README.md`](../../logs/README.md).
