# Prompts

The prompt templates of the synonym classification, in the format of the `model` package (`Model.addPromptFromFile()` and `Model.formatPromptFromFile()`): `{placeholder}` fields, with literal braces written as `{{` and `}}`.

| File | Content |
|---|---|
| `single_prompt.md` | Classifies a synonym as expert or layperson in one prompt |
| `multiple_prompt1.md` | Chain-of-thoughts, step 1: the meaning of the HPO concept |
| `multiple_prompt2.md` | Chain-of-thoughts, step 2: who uses the synonym |
| `multiple_prompt3.md` | Chain-of-thoughts, step 3: the classification |

The answer of `single_prompt.md` and `multiple_prompt3.md` is a JSON object `{"classification": ..., "confidence": ...}`.

## Configuration

`syntype.py` uses the templates of its experiment, set in the `syntype` section of [`config/config.yaml`](../config/config.yaml), from the folder `llm.prompt_folder`:

```yaml
syntype:
  prompt_files:
    single: ["single_prompt.md"]
    chain_of_thoughts: ["multiple_prompt1.md", "multiple_prompt2.md", "multiple_prompt3.md"]
```

It passes them to the `model` package as `llm.prompt_files`, which records their hashes in the run metadata and names the checkpoint columns after them. A template can be edited in place; `syntype.py` then refuses to continue a run made with the old text.

## Placeholders

| Placeholder | Used in | Value |
|---|---|---|
| `{input}` | `single_prompt.md`, `multiple_prompt1.md` | The concept as written by `formatInput()` in `src/syntype.py`: label, and the definition, comment, parents and children the experiment adds; in `single_prompt.md` also the synonym |
| `{synonym}` | `multiple_prompt2.md` | The synonym |
| `{examples}` | `single_prompt.md`, `multiple_prompt3.md` | Two line breaks followed by `getExamples()` for Few-Shot runs, otherwise empty |

`src/syntype.py` fills them in (`sourceClassificationPrompt()` and `sourceClassificationPrompt1()` to `3()`).
