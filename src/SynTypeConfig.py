"""
Pydantic schema of the "syntype" section of config/config.yaml and of its
experiments, validated by config.py.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing   import Literal

class ExperimentConfig(BaseModel):
    """
    One experiment of syntype.experiments: how the synonyms are classified.
    A setting that is not given is off.
    """
    model_config = ConfigDict(extra = "forbid")

    # Only the synonyms of the development set (hpo.development.csv, see
    # syntype.development_set); otherwise those of the evaluation set
    # (hpo.evaluation.csv, the expert and layperson synonyms of hpo.csv
    # without the development synonyms).
    test: bool                                  = False
    # Examples in the prompts.
    few_shot: bool                              = False
    # Three prompts instead of one.
    chain_of_thoughts: bool                     = False
    # Information about the concept added to the prompts: its definition,
    # its comment, and its structure (the labels of its parents and
    # children).
    definition: bool                            = False
    comment: bool                               = False
    structure: bool                             = False


class DevelopmentSetConfig(BaseModel):
    """
    The development set (hpo.development.csv, written by SynTypeDevelopmentSet.py
    and the preparation run): synonyms drawn at random from the transformed
    HPO, to formulate the prompts and to find the mean and standard deviation
    of the cosine distances; runs with "test" classify them.
    """
    model_config = ConfigDict(extra = "forbid")

    # Number of synonyms drawn per gold type (expert, layperson).
    size_per_type: int                          = Field(default = 500, gt = 0)
    # Seed of the random selection, so the same set is drawn again.
    seed: int                                   = 0


class SyntypeConfig(BaseModel):
    """
    The "syntype" section of the configuration file: the settings of syntype
    itself. The settings of the packages it uses are in their own sections
    (adapter, embedder, llm, logger).
    """
    model_config = ConfigDict(extra = "forbid")

    # The experiments, picked by their index on the command line (see
    # start.sh and finalize.sh); at least one.
    experiments: list[ExperimentConfig]         = Field(min_length = 1)

    # The development set ("test").
    development_set: DevelopmentSetConfig       = DevelopmentSetConfig()

    # Folder of the data, relative to the repository root.
    data_folder: str                            = "./data"

    # Name used in the distance column names for every model in
    # embedder.model_id.
    embedding_model_names: dict[str, str]       = {}

    # Mean and standard deviation of the cosine distance between every
    # synonym and the label of its concept, derived from the development set,
    # with which the distances are normalised. The defaults 0 and 1 leave them as
    # they are; None computes them from the data.
    cosine_distance_mean: float | None          = 0.0
    cosine_distance_std: float | None           = Field(default = 1.0, gt = 0)

    # Range of the normalised distances in the evaluation plots and of the
    # thresholds tried.
    distance_evaluation_lower_bound: float      = -4
    distance_evaluation_upper_bound: float      = 4

    # Draw the distance plots of the evaluation also without "test", and
    # again if they exist already.
    force_draw_additional_plots: bool           = False
