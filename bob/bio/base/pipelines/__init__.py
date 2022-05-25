# isort: skip_file
from .pipelines import PipelineSimple  # noqa: F401

import gzip
import pickle
import os


def pickle_compress(obj, path, attempts=5):
    """
    Pickle an object, compressed it and save it

    Parameters
    ----------

       path: str
          Path where to save the object

       obj:
          Object to be saved

       attempts: Serialization attempts

    """
    for i in range(attempts):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Trying to get writting right
            # This might fail in our file system
            with gzip.open(path, "wb") as f:
                f.write(pickle.dumps(obj))

            # Testing unpression
            uncompress_unpickle(path)
            break
        except Exception:
            continue
    else:
        # If it fails in the 5 attemps
        raise EOFError(f"Failed to serialize/desserialize {path}")


def uncompress_unpickle(path):

    with gzip.open(path, "rb") as f:
        return pickle.loads(f.read())


from .score_writers import FourColumnsScoreWriter, CSVScoreWriter
from .wrappers import (
    CheckpointWrapper,
    DaskWrapper,
    checkpoint_pipeline_simple,
    dask_pipeline_simple,
    is_checkpointed,
    get_pipeline_simple_tags,
)

from .abstract_classes import BioAlgorithm, Database, ScoreWriter

from .score_post_processor import (  # noqa: F401
    PipelineScoreNorm,
    ZNormScores,
    TNormScores,
    dask_score_normalization_pipeline,
    CategoricalCalibration,
    WeibullCalibration,
    LLRCalibration,
    GammaCalibration,
    BetaCalibration,
)

from .entry_points import (  # noqa: F401
    execute_pipeline_simple,
    execute_pipeline_score_norm,
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
    CheckpointWrapper,
    DaskWrapper,
    dask_pipeline_simple,
    checkpoint_pipeline_simple,
    is_checkpointed,
    get_pipeline_simple_tags,
    execute_pipeline_simple,
    BioAlgorithm,
    Database,
    ScoreWriter,
)

__all__ = [_ for _ in dir() if not _.startswith("_")]
