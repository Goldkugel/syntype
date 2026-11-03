from pydantic import BaseModel, ConfigDict, Field
from typing   import Literal

# Similarity metrics syntype.py can compute between a synonym and its label.
SimilarityMetric = Literal["cosine", "euclidean", "manhattan", "angular", "mahalanobis"]


class ExperimentConfig(BaseModel):
    """
    The experiment a run performs if it is started without words after the
    model ID. Otherwise these words (see start.sh), e.g. "few-shot", set
    every setting.
    """

    model_config = ConfigDict(extra = "forbid")

    # Only the test concepts (testIDs in config.py) and their parents and children
    # ("test").
    test: bool              = False
    # Examples in the prompts ("few-shot").
    few_shot: bool          = False
    # Three prompts instead of one ("chain-of-thoughts").
    chain_of_thoughts: bool = False
    # Information about the concept added to the prompts ("definition",
    # "comment", "parents", "children").
    definition: bool        = False
    comment: bool           = False
    parents: bool           = False
    children: bool          = False


class ProgressBarConfig(BaseModel):
    """
    Look of the progress bars on the console.
    """

    model_config = ConfigDict(extra = "forbid")

    color: str       = "cyan"
    text_length: int = Field(default = 40, gt = 0)


class PromptFilesConfig(BaseModel):
    """
    Prompt templates of the classification (see prompt/README.md), one per
    turn of the conversation.
    """

    model_config = ConfigDict(extra = "forbid")

    single: list[str]            = Field(
        default = ["single_prompt.md"], min_length = 1, max_length = 1)
    chain_of_thoughts: list[str] = Field(
        default = ["multiple_prompt1.md", "multiple_prompt2.md", "multiple_prompt3.md"],
        min_length = 3, max_length = 3)


class SyntypeConfig(BaseModel):
    """
    The "syntype" section of the configuration file: the settings of syntype
    itself. The settings of the packages it uses are in their own sections
    (adapter, embedder, llm, logger).
    """

    model_config = ConfigDict(extra = "forbid")

    experiment: ExperimentConfig     = ExperimentConfig()

    # Folder of the data, relative to src/.
    data_folder: str                 = "../data"

    # Name used in the similarity column names for every model in
    # embedder.model_id.
    embedding_model_names: dict[str, str] = {}

    # Similarities syntype.py computes between every synonym and its label.
    similarity_metrics: list[SimilarityMetric] = []

    # Range of the normalised similarities in the evaluation plots.
    similarity_evaluation_lower_bound: float = -4
    similarity_evaluation_upper_bound: float = 4

    # Draw the similarity plots of the evaluation also without "test", and
    # again if they exist already.
    force_draw_additional_plots: bool = False

    progress_bar: ProgressBarConfig  = ProgressBarConfig()

    prompt_files: PromptFilesConfig  = PromptFilesConfig()
