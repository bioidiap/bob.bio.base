from .pipelines import VanillaBiometricsPipeline

import pickle
import gzip

import os


def pickle_compress(path, obj, attempts=5):
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
        except:
            continue
    else:
        # If it fails in the 5 attemps
        raise EOFError(f"Failed to serialize/desserialize {path}")


def uncompress_unpickle(path):

    with gzip.open(path, "rb") as f:
        return pickle.loads(f.read())


from .biometric_algorithms import Distance
from .score_writers import FourColumnsScoreWriter, CSVScoreWriter
from .wrappers import (
    BioAlgorithmCheckpointWrapper,
    BioAlgorithmDaskWrapper,
    dask_vanilla_biometrics,
    checkpoint_vanilla_biometrics,
    is_checkpointed,
)

from .abstract_classes import BioAlgorithm, Database, ScoreWriter

from .zt_norm import ZTNormPipeline, ZTNormDaskWrapper, ZTNormCheckpointWrapper, ZTNorm

from .legacy import BioAlgorithmLegacy, DatabaseConnector

from .vanilla_biometrics import (
    execute_vanilla_biometrics,
    execute_vanilla_biometrics_ztnorm,
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
    VanillaBiometricsPipeline,
    Distance,
    FourColumnsScoreWriter,
    CSVScoreWriter,
    BioAlgorithmCheckpointWrapper,
    BioAlgorithmDaskWrapper,
    dask_vanilla_biometrics,
    checkpoint_vanilla_biometrics,
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

