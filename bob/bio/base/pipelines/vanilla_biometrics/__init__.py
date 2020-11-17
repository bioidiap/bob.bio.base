from .pipelines import VanillaBiometricsPipeline

from .biometric_algorithms import Distance
from .score_writers import FourColumnsScoreWriter, CSVScoreWriter
from .wrappers import (
    BioAlgorithmCheckpointWrapper,
    BioAlgorithmDaskWrapper,
    dask_vanilla_biometrics,
    checkpoint_vanilla_biometrics,
    dask_get_partition_size,
    is_checkpointed,
)

from .abstract_classes import BioAlgorithm, Database, ScoreWriter

from .zt_norm import ZTNormPipeline, ZTNormDaskWrapper, ZTNormCheckpointWrapper, ZTNorm

from .legacy import BioAlgorithmLegacy, DatabaseConnector

from .vanilla_biometrics import execute_vanilla_biometrics


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
    VanillaBiometricsPipeline,
    Distance,
    FourColumnsScoreWriter,
    CSVScoreWriter,
    BioAlgorithmCheckpointWrapper,
    BioAlgorithmDaskWrapper,
    dask_vanilla_biometrics,
    checkpoint_vanilla_biometrics,
    dask_get_partition_size,
    is_checkpointed,
    ZTNormPipeline,
    ZTNormDaskWrapper,
    ZTNormCheckpointWrapper,
    BioAlgorithmLegacy,
    DatabaseConnector,
    execute_vanilla_biometrics,
    BioAlgorithm,
    Database,
    ScoreWriter,
    ZTNorm,
)

__all__ = [_ for _ in dir() if not _.startswith("_")]

