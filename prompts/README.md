# Prompts

The prompt templates of the synonym classification, in the format of the `model` package (`Model.addPromptFromFile()` and `Model.formatPromptFromFile()`): `{placeholder}` fields, with literal braces written as `{{` and `}}`.

| File | Content |
|---|---|
| `single_prompt.md` | Classifies a synonym as expert or layperson in one prompt |
| `multiple_prompt1.md` | Chain-of-thoughts, step 1: the meaning of the HPO concept |
| `multiple_prompt2.md` | Chain-of-thoughts, step 2: who uses the synonym |
| `multiple_prompt3.md` | Chain-of-thoughts, step 3: the classification |

The answer of `single_prompt.md` and `multiple_prompt3.md` is the same JSON object `{"classification": ..., "confidence": ..., "rationale": ...}`, with a confidence from 1 to 5.

## Configuration

The templates are listed in `llm.prompt_files` of [`config/config.yaml`](../config/config.yaml), in the folder `llm.prompt_folder`, and used by their index: 0 is the single prompt, 1 to 3 are the chain-of-thoughts prompts.

```yaml
llm:
  prompt_folder: "./prompts/"
  prompt_files:
    - "single_prompt.md"
    - "multiple_prompt1.md"
    - "multiple_prompt2.md"
    - "multiple_prompt3.md"
```

The `model` package records the hashes of all of them in the run metadata. A template can be edited in place: the prompts made from it are new, so their answers are generated anew, while the answers to the unchanged prompts are taken from the model's answer store.

## Placeholders

| Placeholder | Used in | Value |
|---|---|---|
| `{hpo_label}` | `single_prompt.md`, `multiple_prompt1.md` | The label of the concept |
| `{hpo_synonyms}` | `single_prompt.md`, `multiple_prompt2.md` | The synonym to classify |
| `{hpo_definition}`, `{hpo_comment}`, `{hpo_parents}`, `{hpo_children}` | `single_prompt.md`, `multiple_prompt1.md` | The definition, comment, parents and children of the concept; empty if the experiment does not add it |
| `{examples}` | `single_prompt.md`, `multiple_prompt3.md` | Two line breaks followed by `getExamples()` for Few-Shot runs, otherwise empty; the examples are written in the same input format |

The `model` package loads a template and fills it in for every synonym (`Model.addPromptsFromFile()`) with the values of `promptValues()` in `src/syntype.py`, which also gives the order of the templates: index 0, or 1 to 3 for chain-of-thoughts.
