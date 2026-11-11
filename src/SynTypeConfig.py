"""
Pydantic schema of the "syntype" section of config/config.yaml and of its
experiments, validated by config.py.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

class ExperimentConfig(BaseModel):
    """
    One experiment of syntype.experiments: how the synonyms are classified.
    A setting that is not given is off.
    """
    model_config = ConfigDict(extra = "forbid")

    # Only the synonyms of the development set (hpo.development.csv);
    # otherwise those of the evaluation set (hpo.evaluation.csv).
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
    The development set (hpo.development.csv): synonyms drawn at random, to
    formulate the prompts and to find the mean and standard deviation of the
    cosine distances and the distance threshold; runs with "test" classify
    them. Drawn together with the evaluation set (hpo.evaluation.csv).
    """
    model_config = ConfigDict(extra = "forbid")

    # Number of synonyms drawn per gold type (expert, layperson).
    size_per_type: int                          = Field(default = 500, gt = 0)
    # Seed of the random selection, so the same set is drawn again.
    seed: int                                   = 0


class SyntypeConfig(BaseModel):
    """
    The "syntype" section of the configuration file; the packages it uses have
    their own sections (adapter, embedder, llm, logger).
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
    # embedder.model_id; by default its ID without the organisation.
    embedding_model_names: dict[str, str]       = {}

    # Mean and standard deviation of the synonym-label cosine distance of the
    # development set, to normalise the distances. The defaults 0 and 1 leave
    # them as they are; None computes them from the data.
    cosine_distance_mean: float | None          = 0.0
    cosine_distance_std: float | None           = Field(default = 1.0, gt = 0)

    # Range of the normalised distances in the evaluation plots and of the
    # thresholds tried.
    distance_evaluation_lower_bound: float      = -4
    distance_evaluation_upper_bound: float      = 4
    # Distance between two thresholds tried.
    distance_evaluation_step: float             = Field(default = 0.001, gt = 0)

    # The threshold of the normalised cosine distance per embedding model
    # (named as in the distance columns), evaluated as the model "<model>
    # (distance)": layperson at or above it, expert below. SynTypeEval.py logs
    # the best threshold on the development set.
    distance_thresholds: dict[str, float]       = {}

    # Bootstrap samples of the 95% intervals in the evaluation summary.
    bootstrap_samples: int                      = Field(default = 1000, gt = 0)

    # Draw the distance plots of the evaluation also without "test".
    force_draw_additional_plots: bool           = False

    @model_validator(mode = "after")
    def checkDistanceBounds(self) -> "SyntypeConfig":
        """
        The lower bound of the distance evaluation must be below the upper one.
        """
        if self.distance_evaluation_lower_bound >= self.distance_evaluation_upper_bound:
            raise ValueError("distance_evaluation_lower_bound must be below " \
                "distance_evaluation_upper_bound.")
        return self
