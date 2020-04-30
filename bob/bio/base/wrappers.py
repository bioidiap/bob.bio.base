from bob.bio.base.transformers import (
    CheckpointPreprocessor,
    CheckpointExtractor,
    CheckpointAlgorithm,
)
from bob.bio.base.preprocessor import Preprocessor
from bob.bio.base.extractor import Extractor
from bob.bio.base.algorithm import Algorithm
import os


def wrap_bob_bio_legacy(instance, dir_name, **kwargs):
    """
    Wraps either :any:`bob.bio.base.preprocessor.Preprocessor`,
    :any:`bob.bio.base.extractor.Extractor`, or :any:`bob.bio.base.algorithm.Algorithm`
    with :any:`sklearn.base.TransformerMixin` and :any:`bob.pipelines.CheckpointWrapper`
    and :any:`bob.pipelines.SampleWrapper`


    Parameters
    ----------
    instance : object
        Instance of :any:`bob.bio.base.preprocessor.Preprocessor`,
        :any:`bob.bio.base.extractor.Extractor`, or
        :any:`bob.bio.base.algorithm.Algorithm`
    dir_name : str
        Directory name for the checkpoints
    **kwargs
        All extra arguments are passed to
        :any:`CheckpointPreprocessor`, :any:`CheckpointExtractor`, or
        :any:`CheckpointAlgorithm` depending on the instance.

    Returns
    -------
    object
        The wrapped instance.

    Raises
    ------
    ValueError
        If instance is not one of Preprocessor, Extractor, or Algorithm.
    """
    model_path = None
    if isinstance(instance, Preprocessor):
        features_dir = os.path.join(dir_name, "preprocessed")
        klass = CheckpointPreprocessor
    elif isinstance(instance, Extractor):
        features_dir = os.path.join(dir_name, "extracted")
        model_path = os.path.join(dir_name, "Extractor.hdf5")
        klass = CheckpointExtractor
    elif isinstance(instance, Algorithm):
        features_dir = os.path.join(dir_name, "projected")
        model_path = os.path.join(dir_name, "Projector.hdf5")
        klass = CheckpointAlgorithm
    else:
        raise ValueError(
            "`instance` should be an instance of `Preprocessor`, `Extractor`, or `Algorithm`"
        )

    instance = klass(
        instance, features_dir=features_dir, model_path=model_path, **kwargs
    )
    return instance
