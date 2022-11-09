# isort: skip_file
from .pipelines import PipelineSimple  # noqa: F401
from .score_writers import FourColumnsScoreWriter, CSVScoreWriter
from .wrappers import (  # noqa: F401
    BioAlgCheckpointWrapper,
    BioAlgDaskWrapper,
    checkpoint_pipeline_simple,
    dask_bio_pipeline,
    is_biopipeline_checkpointed,
    get_bio_alg_tags,
)

from .abstract_classes import BioAlgorithm, Database, ScoreWriter

from .score_post_processor import (  # noqa: F401
    PipelineScoreNorm,
    ZNormScores,
    TNormScores,
    CategoricalCalibration,
    WeibullCalibration,
    LLRCalibration,
    GammaCalibration,
    BetaCalibration,
)

from .entry_points import (  # noqa: F401
    execute_pipeline_simple,
    execute_pipeline_score_norm,
    execute_pipeline_train,
)


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        An iterable of objects to modify

    Resolves `Sphinx referencing issues
    <https://github.com/sphinx-doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    PipelineSimple,
    FourColumnsScoreWriter,
    CSVScoreWriter,
    BioAlgCheckpointWrapper,
    BioAlgDaskWrapper,
    BioAlgorithm,
    Database,
    ScoreWriter,
    PipelineScoreNorm,
    ZNormScores,
    TNormScores,
    CategoricalCalibration,
    WeibullCalibration,
    LLRCalibration,
    GammaCalibration,
    BetaCalibration,
)

__all__ = [_ for _ in dir() if not _.startswith("_")]
