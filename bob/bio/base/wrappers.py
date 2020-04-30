#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.transformers import (
    PreprocessorTransformer,
    ExtractorTransformer,
    AlgorithmTransformer,
)
from bob.bio.base.preprocessor import Preprocessor
from bob.bio.base.extractor import Extractor
from bob.bio.base.algorithm import Algorithm
import bob.pipelines as mario
import os



def wrap_transform_bob(
    bob_object, dir_name, fit_extra_arguments=None, transform_extra_arguments=None
):
    """
    Wraps either :any:`bob.bio.base.preprocessor.Preprocessor`, :any:`bob.bio.base.extractor.Extractor`
    or :any:`bob.bio.base.algorithm.Algorithm` with :any:`sklearn.base.TransformerMixin` 
    and :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`


    Parameters
    ----------

    bob_object: object
        Instance of :any:`bob.bio.base.preprocessor.Preprocessor`, :any:`bob.bio.base.extractor.Extractor` and :any:`bob.bio.base.algorithm.Algorithm`
        
    dir_name: str
        Directory name for the checkpoints

    fit_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.fit_extra_arguments`

    transform_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`


    """

    if isinstance(bob_object, Preprocessor):
        preprocessor_transformer = PreprocessorTransformer(bob_object)
        return wrap_preprocessor(
            preprocessor_transformer,
            features_dir=os.path.join(dir_name, "preprocessor"),
            transform_extra_arguments=transform_extra_arguments,
        )
    elif isinstance(bob_object, Extractor):
        extractor_transformer = ExtractorTransformer(bob_object)
        path = os.path.join(dir_name, "extractor")
        return wrap_extractor(
            extractor_transformer,
            features_dir=path,
            model_path=os.path.join(path, "extractor.pkl"),
            transform_extra_arguments=transform_extra_arguments,
            fit_extra_arguments=fit_extra_arguments,
        )
    elif isinstance(bob_object, Algorithm):
        path = os.path.join(dir_name, "algorithm")
        algorithm_transformer = AlgorithmTransformer(
            bob_object, projector_file=os.path.join(path, "Projector.hdf5")
        )
        return wrap_algorithm(
            algorithm_transformer,
            features_dir=path,
            model_path=os.path.join(path, "algorithm.pkl"),
            transform_extra_arguments=transform_extra_arguments,
            fit_extra_arguments=fit_extra_arguments,
        )
    else:
        raise ValueError(
            "`bob_object` should be an instance of `Preprocessor`, `Extractor` and `Algorithm`"
        )


def wrap_preprocessor(
    preprocessor_transformer, features_dir=None, transform_extra_arguments=None,
):
    """
    Wraps :any:`bob.bio.base.transformers.PreprocessorTransformer` with 
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`

    Parameters
    ----------

    preprocessor_transformer: :any:`bob.bio.base.transformers.PreprocessorTransformer`
       Instance of :any:`bob.bio.base.transformers.PreprocessorTransformer` to be wrapped

    features_dir: str
       Features directory to be checkpointed

    transform_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`

    """

    if not isinstance(preprocessor_transformer, PreprocessorTransformer):
        raise ValueError(
            f"Expected an  instance of PreprocessorTransformer, not {preprocessor_transformer}"
        )

    return mario.wrap(
        ["sample", "checkpoint"],
        preprocessor_transformer,
        load_func=preprocessor_transformer.callable.read_data,
        save_func=preprocessor_transformer.callable.write_data,
        features_dir=features_dir,
        transform_extra_arguments=transform_extra_arguments,
    )


def wrap_extractor(
    extractor_transformer,
    fit_extra_arguments=None,
    transform_extra_arguments=None,
    features_dir=None,
    model_path=None,
):
    """
    Wraps :any:`bob.bio.base.transformers.ExtractorTransformer` with 
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`

    Parameters
    ----------

    extractor_transformer: :any:`bob.bio.base.transformers.ExtractorTransformer`
       Instance of :any:`bob.bio.base.transformers.ExtractorTransformer` to be wrapped

    features_dir: str
       Features directory to be checkpointed

    model_path: str
       Path to checkpoint the model

    fit_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.fit_extra_arguments`

    transform_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`

    """

    if not isinstance(extractor_transformer, ExtractorTransformer):
        raise ValueError(
            f"Expected an  instance of ExtractorTransformer, not {extractor_transformer}"
        )

    return mario.wrap(
        ["sample", "checkpoint"],
        extractor_transformer,
        load_func=extractor_transformer.callable.read_feature,
        save_func=extractor_transformer.callable.write_feature,
        model_path=model_path,
        features_dir=features_dir,
        transform_extra_arguments=transform_extra_arguments,
        fit_extra_arguments=fit_extra_arguments,
    )


def wrap_algorithm(
    algorithm_transformer,
    fit_extra_arguments=None,
    transform_extra_arguments=None,
    features_dir=None,
    model_path=None,
):
    """
    Wraps :any:`bob.bio.base.transformers.AlgorithmTransformer` with 
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`

    Parameters
    ----------

    algorithm_transformer: :any:`bob.bio.base.transformers.AlgorithmTransformer`
       Instance of :any:`bob.bio.base.transformers.AlgorithmTransformer` to be wrapped

    features_dir: str
       Features directory to be checkpointed

    model_path: str
       Path to checkpoint the model

    fit_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.fit_extra_arguments`

    transform_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`

    """

    if not isinstance(algorithm_transformer, AlgorithmTransformer):
        raise ValueError(
            f"Expected an  instance of AlgorithmTransformer, not {algorithm_transformer}"
        )

    return mario.wrap(
        ["sample", "checkpoint"],
        algorithm_transformer,
        load_func=algorithm_transformer.callable.read_feature,
        save_func=algorithm_transformer.callable.write_feature,
        model_path=model_path,
        features_dir=features_dir,
        transform_extra_arguments=transform_extra_arguments,
        fit_extra_arguments=fit_extra_arguments,
    )
