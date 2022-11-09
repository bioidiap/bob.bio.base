#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os

import bob.pipelines

from bob.bio.base.extractor import Extractor
from bob.bio.base.preprocessor import Preprocessor
from bob.bio.base.transformers import (
    ExtractorTransformer,
    PreprocessorTransformer,
)
from bob.bio.base.utils import is_argument_available


def wrap_bob_legacy(
    bob_object,
    dir_name,
    fit_extra_arguments=None,
    transform_extra_arguments=None,
    dask_it=False,
    **kwargs,
):
    """
    Wraps either :any:`bob.bio.base.preprocessor.Preprocessor` or
    :any:`bob.bio.base.extractor.Extractor` with
    :any:`sklearn.base.TransformerMixin` and
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and
    :any:`bob.pipelines.wrappers.SampleWrapper`


    Parameters
    ----------

    bob_object: object
        Instance of :any:`bob.bio.base.preprocessor.Preprocessor` or
        :any:`bob.bio.base.extractor.Extractor`

    dir_name: str
        Directory name for the checkpoints

    fit_extra_arguments: [tuple]
        Same behavior as in Check
        :any:`bob.pipelines.wrappers.fit_extra_arguments`

    transform_extra_arguments: [tuple]
        Same behavior as in Check
        :any:`bob.pipelines.wrappers.transform_extra_arguments`

    dask_it: bool
        If True, the transformer will be a dask graph
    """

    if isinstance(bob_object, Preprocessor):
        transformer = wrap_checkpoint_preprocessor(
            bob_object,
            features_dir=os.path.join(dir_name, "preprocessor"),
            **kwargs,
        )
    elif isinstance(bob_object, Extractor):
        transformer = wrap_checkpoint_extractor(
            bob_object,
            features_dir=os.path.join(dir_name, "extractor"),
            model_path=dir_name,
            fit_extra_arguments=fit_extra_arguments,
            transform_extra_arguments=transform_extra_arguments,
            **kwargs,
        )
    else:
        raise ValueError(
            "`bob_object` should be an instance of `Preprocessor`, `Extractor` and `Algorithm`"
        )

    if dask_it:
        transformer = bob.pipelines.wrap(["dask"], transformer)

    return transformer


def wrap_sample_preprocessor(
    preprocessor,
    transform_extra_arguments=(("annotations", "annotations"),),
    **kwargs,
):
    """
    Wraps :any:`bob.bio.base.preprocessor.Preprocessor` with
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`

    .. warning::
       This wrapper doesn't checkpoint data

    Parameters
    ----------

    preprocessor: :any:`bob.bio.base.preprocessor.Preprocessor`
       Instance of :any:`bob.bio.base.transformers.PreprocessorTransformer` to be wrapped

    transform_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`

    """

    transformer = PreprocessorTransformer(preprocessor)
    return bob.pipelines.wrap(
        ["sample"],
        transformer,
        transform_extra_arguments=transform_extra_arguments,
    )


def wrap_checkpoint_preprocessor(
    preprocessor,
    features_dir=None,
    transform_extra_arguments=(("annotations", "annotations"),),
    load_func=None,
    save_func=None,
    extension=".hdf5",
):
    """
    Wraps :any:`bob.bio.base.preprocessor.Preprocessor` with
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`

    Parameters
    ----------

    preprocessor: :any:`bob.bio.base.preprocessor.Preprocessor`
       Instance of :any:`bob.bio.base.transformers.PreprocessorTransformer` to be wrapped

    features_dir: str
       Features directory to be checkpointed (see :any:bob.pipelines.CheckpointWrapper`).

    extension : str, optional
        Extension o preprocessed files (see :any:bob.pipelines.CheckpointWrapper`).

    load_func : None, optional
        Function that loads data to be preprocessed.
        The default is :any:`bob.bio.base.preprocessor.Preprocessor.read_data`

    save_func : None, optional
        Function that saves preprocessed data.
        The default is :any:`bob.bio.base.preprocessor.Preprocessor.write_data`

    transform_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`

    """

    transformer = PreprocessorTransformer(preprocessor)
    return bob.pipelines.wrap(
        ["sample", "checkpoint"],
        transformer,
        load_func=load_func or preprocessor.read_data,
        save_func=save_func or preprocessor.write_data,
        features_dir=features_dir,
        transform_extra_arguments=transform_extra_arguments,
        extension=extension,
    )


def _prepare_extractor_sample_args(
    extractor, transform_extra_arguments, fit_extra_arguments
):
    if transform_extra_arguments is None and is_argument_available(
        "metadata", extractor.__call__
    ):
        transform_extra_arguments = (("metadata", "metadata"),)

    if (
        fit_extra_arguments is None
        and extractor.requires_training
        and extractor.split_training_data_by_client
    ):
        fit_extra_arguments = (("y", "subject"),)

    return transform_extra_arguments, fit_extra_arguments


def wrap_sample_extractor(
    extractor,
    fit_extra_arguments=None,
    transform_extra_arguments=None,
    model_path=None,
    **kwargs,
):
    """
    Wraps :any:`bob.bio.base.extractor.Extractor` with
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`

    Parameters
    ----------

    extractor: :any:`bob.bio.base.extractor.Preprocessor`
       Instance of :any:`bob.bio.base.transformers.ExtractorTransformer` to be wrapped

    transform_extra_arguments: [tuple], optional
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`

    model_path: str
        Path to `extractor_file` in :any:`bob.bio.base.extractor.Extractor`

    """

    extractor_file = (
        os.path.join(model_path, "Extractor.hdf5")
        if model_path is not None
        else None
    )

    transformer = ExtractorTransformer(extractor, model_path=extractor_file)

    (
        transform_extra_arguments,
        fit_extra_arguments,
    ) = _prepare_extractor_sample_args(
        extractor, transform_extra_arguments, fit_extra_arguments
    )

    return bob.pipelines.wrap(
        ["sample"],
        transformer,
        transform_extra_arguments=transform_extra_arguments,
        fit_extra_arguments=fit_extra_arguments,
        **kwargs,
    )


def wrap_checkpoint_extractor(
    extractor,
    features_dir=None,
    fit_extra_arguments=None,
    transform_extra_arguments=None,
    load_func=None,
    save_func=None,
    extension=".hdf5",
    model_path=None,
    **kwargs,
):
    """
    Wraps :any:`bob.bio.base.extractor.Extractor` with
    :any:`bob.pipelines.wrappers.CheckpointWrapper` and :any:`bob.pipelines.wrappers.SampleWrapper`

    Parameters
    ----------

    extractor: :any:`bob.bio.base.extractor.Preprocessor`
       Instance of :any:`bob.bio.base.transformers.ExtractorTransformer` to be wrapped

    features_dir: str
       Features directory to be checkpointed (see :any:bob.pipelines.CheckpointWrapper`).

    extension : str, optional
        Extension o preprocessed files (see :any:bob.pipelines.CheckpointWrapper`).

    load_func : None, optional
        Function that loads data to be preprocessed.
        The default is :any:`bob.bio.base.extractor.Extractor.read_feature`

    save_func : None, optional
        Function that saves preprocessed data.
        The default is :any:`bob.bio.base.extractor.Extractor.write_feature`

    fit_extra_arguments: [tuple]
        Same behavior as in Check :any:`bob.pipelines.wrappers.fit_extra_arguments`

    transform_extra_arguments: [tuple], optional
        Same behavior as in Check :any:`bob.pipelines.wrappers.transform_extra_arguments`

    model_path: str
        See :any:`TransformerExtractor`.

    """

    extractor_file = (
        os.path.join(model_path, "Extractor.hdf5")
        if model_path is not None
        else None
    )

    model_file = (
        os.path.join(model_path, "Extractor.pkl")
        if model_path is not None
        else None
    )
    transformer = ExtractorTransformer(extractor, model_path=extractor_file)

    (
        transform_extra_arguments,
        fit_extra_arguments,
    ) = _prepare_extractor_sample_args(
        extractor, transform_extra_arguments, fit_extra_arguments
    )

    return bob.pipelines.wrap(
        ["sample", "checkpoint"],
        transformer,
        load_func=load_func or extractor.read_feature,
        save_func=save_func or extractor.write_feature,
        model_path=model_file,
        features_dir=features_dir,
        transform_extra_arguments=transform_extra_arguments,
        fit_extra_arguments=fit_extra_arguments,
        **kwargs,
    )
