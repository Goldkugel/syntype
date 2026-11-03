from pydantic import BaseModel, ConfigDict, Field
from typing   import Literal

class SyntypeConfig(BaseModel):
    """
    The "syntype" section of the configuration file: the settings of syntype
    itself. The settings of the packages it uses are in their own sections
    (adapter, embedder, llm, logger).
    """
    model_config = ConfigDict(extra = "forbid")

    # Only the test concepts (testIDs in config.py) and their parents and 
    # children ("test").
    test: bool                                  = False
    # Examples in the prompts ("few-shot").
    few_shot: bool                              = False
    # Three prompts instead of one ("chain-of-thoughts").
    chain_of_thoughts: bool                     = False
    # Information about the concept added to the prompts ("definition",
    # "comment", "parents", "children").
    definition: bool                            = False
    comment: bool                               = False
    parents: bool                               = False
    children: bool                              = False

    # Folder of the data, relative to src/.
    data_folder: str                            = "./data"

    # Name used in the similarity column names for every model in
    # embedder.model_id.
    embedding_model_names: dict[str, str]       = {}

    # Range of the normalised similarities in the evaluation plots.
    similarity_evaluation_lower_bound: float    = -4
    similarity_evaluation_upper_bound: float    = 4

    # Draw the similarity plots of the evaluation also without "test", and
    # again if they exist already.
    force_draw_additional_plots: bool           = False
