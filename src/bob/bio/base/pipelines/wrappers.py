import functools
import logging
import os

from typing import Any, Callable, Optional

import dask
import h5py
import numpy as np

from sklearn.pipeline import Pipeline

import bob.pipelines

from bob.bio.base.pipelines import PipelineSimple
from bob.pipelines import (
    CheckpointWrapper,
    DelayedSample,
    Sample,
    is_instance_nested,
)
from bob.pipelines.wrappers import BaseWrapper, _frmt, get_bob_tags

from .abstract_classes import BioAlgorithm

logger = logging.getLogger(__name__)


def default_save(data: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f["data"] = data


def default_load(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["data"][()]


def get_bio_alg_tags(estimator=None, force_tags=None):
    bob_tags = get_bob_tags(estimator=estimator, force_tags=force_tags)
    default_tags = {
        "bob_enrolled_extension": ".h5",
        "bob_enrolled_save_fn": default_save,
        "bob_enrolled_load_fn": default_load,
    }
    force_tags = force_tags or {}
    estimator_tags = estimator._get_tags() if estimator is not None else {}
    return {**bob_tags, **default_tags, **estimator_tags, **force_tags}


class BioAlgorithmBaseWrapper(BioAlgorithm, BaseWrapper):
    def create_templates(self, feature_sets, enroll):
        return self.biometric_algorithm.create_templates(feature_sets, enroll)

    def compare(self, enroll_templates, probe_templates):
        return self.biometric_algorithm.compare(
            enroll_templates, probe_templates
        )


class BioAlgCheckpointWrapper(BioAlgorithmBaseWrapper):
    """Wrapper used to checkpoint enrolled and Scoring samples.

    Parameters
    ----------
    biometric_algorithm
        An implemented :any:`BioAlgorithm`

    base_dir
        Path to store biometric references and scores

    extension
        Default extension of the enrolled references files.
        If None, will use the ``bob_checkpoint_extension`` tag in the estimator, or
        default to ``.h5``.

    save_func
        Pointer to a customized function that saves an enrolled reference to the disk.
        If None, will use the ``bob_enrolled_save_fn`` tag in the estimator, or default
        to h5py.

    load_func
        Pointer to a customized function that loads an enrolled reference from disk.
        If None, will use the ``bob_enrolled_load_fn`` tag in the estimator, or default
        to h5py.

    group
        group of the data, used to save different group's checkpoints in different dirs.

    force
        If True, will recompute scores and biometric references no matter if a file
        exists

    hash_fn
        Pointer to a hash function. This hash function maps
        `sample.key` to a hash code and this hash code corresponds a relative directory
        where a single `sample` will be checkpointed.
        This is useful when is desirable file directories with less than a certain
        number of files.

    Examples
    --------

    >>> from bob.bio.base.algorithm import Distance
    >>> from bob.bio.base.pipelines import BioAlgCheckpointWrapper
    >>> biometric_algorithm = BioAlgCheckpointWrapper(Distance(), base_dir="./")
    >>> biometric_algorithm.create_templates(samples, enroll=True) # doctest: +SKIP

    """

    def __init__(
        self,
        biometric_algorithm: BioAlgorithm,
        base_dir: str,
        extension: Optional[str] = None,
        save_func: Optional[Callable[[str, Any], None]] = None,
        load_func: Optional[Callable[[str], Any]] = None,
        group: Optional[str] = None,
        force: bool = False,
        hash_fn: Optional[Callable[[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_dir = base_dir
        self.set_score_references_path(group)
        self.group = group
        self.biometric_algorithm = biometric_algorithm
        self.force = force
        self.hash_fn = hash_fn
        bob_tags = get_bio_alg_tags(self.biometric_algorithm)
        self.extension = extension or bob_tags["bob_enrolled_extension"]
        self.save_func = save_func or bob_tags["bob_enrolled_save_fn"]
        self.load_func = load_func or bob_tags["bob_enrolled_load_fn"]

    def set_score_references_path(self, group):
        if group is None:
            self.biometric_reference_dir = os.path.join(
                self.base_dir, "biometric_references"
            )
        else:
            self.biometric_reference_dir = os.path.join(
                self.base_dir, group, "biometric_references"
            )

    def write_biometric_reference(self, sample, path):
        data = sample.data
        if data is None:
            raise RuntimeError("Cannot checkpoint template of None")
        return self.save_func(sample.data, path)

    def _enroll_sample_set(self, sampleset):
        """
        Enroll a sample set with checkpointing
        """

        # If sampleset has a key use it, otherwise use the first sample's key
        model_key = getattr(sampleset, "key", sampleset.samples[0].key)

        # Amending `models` directory
        hash_dir_name = (
            self.hash_fn(str(model_key)) if self.hash_fn is not None else ""
        )

        path = os.path.join(
            self.biometric_reference_dir,
            hash_dir_name,
            str(model_key) + self.extension,
        )

        if self.force or not os.path.exists(path):
            enrolled_sample = (
                self.biometric_algorithm.create_templates_from_samplesets(
                    [sampleset], enroll=True
                )[0]
            )

            # saving the new sample
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.write_biometric_reference(enrolled_sample, path)

        # This seems inefficient, but it's crucial for large datasets
        delayed_enrolled_sample = DelayedSample(
            functools.partial(self.load_func, path), parent=sampleset
        )

        return delayed_enrolled_sample

    def create_templates_from_samplesets(self, list_of_samplesets, enroll):
        logger.debug(
            f"{_frmt(self, attr='biometric_algorithm')}.create_templates_from_samplesets(... enroll={enroll})"
        )
        if not enroll:
            return self.biometric_algorithm.create_templates_from_samplesets(
                list_of_samplesets, enroll
            )
        retval = []
        for sampleset in list_of_samplesets:
            # if it exists, load it!
            sample = self._enroll_sample_set(sampleset)
            retval.append(sample)
        return retval


class BioAlgDaskWrapper(BioAlgorithmBaseWrapper):
    """
    Wrap :any:`bob.bio.base.pipelines.BioAlgorithm` to work with DASK
    """

    def __init__(self, biometric_algorithm: BioAlgorithm, **kwargs):
        self.biometric_algorithm = biometric_algorithm

    def create_templates_from_samplesets(self, list_of_samplesets, enroll):
        logger.debug(
            f"{_frmt(self, attr='biometric_algorithm')}.create_templates_from_samplesets(... enroll={enroll})"
        )
        templates = list_of_samplesets.map_partitions(
            self.biometric_algorithm.create_templates_from_samplesets,
            enroll=enroll,
        )
        return templates

    def score_sample_templates(
        self, probe_samples, enroll_samples, score_all_vs_all
    ):
        logger.debug(
            f"{_frmt(self, attr='biometric_algorithm')}.score_sample_templates(... score_all_vs_all={score_all_vs_all})"
        )
        # load the templates into memory because they could be delayed samples
        enroll_samples = enroll_samples.map_partitions(
            _delayed_samples_to_samples
        )
        probe_samples = probe_samples.map_partitions(
            _delayed_samples_to_samples
        )

        all_references = dask.delayed(list)(enroll_samples)
        scores = probe_samples.map_partitions(
            self.biometric_algorithm.score_sample_templates,
            all_references,
            score_all_vs_all=score_all_vs_all,
        )
        return scores


def _delayed_samples_to_samples(delayed_samples):
    return [Sample(sample.data, parent=sample) for sample in delayed_samples]


def dask_bio_pipeline(
    pipeline: PipelineSimple,
    npartitions: Optional[int] = None,
    partition_size: Optional[int] = None,
):
    """
    Given a :any:`PipelineSimple`, wraps their :attr:`transformer` and
    :attr:`biometric_algorithm` to be executed with dask.

    Parameters
    ----------

    pipeline
       pipeline to be dasked

    npartitions
       Number of partitions for the initial `dask.bag`

    partition_size
       Size of the partition for the initial `dask.bag`
    """
    dask_wrapper_kw = {}
    if partition_size is None:
        dask_wrapper_kw["npartitions"] = npartitions
    else:
        dask_wrapper_kw["partition_size"] = partition_size

    pipeline.transformer = bob.pipelines.wrap(
        ["dask"], pipeline.transformer, **dask_wrapper_kw
    )
    pipeline.biometric_algorithm = BioAlgDaskWrapper(
        pipeline.biometric_algorithm
    )

    def _write_scores(scores):
        return scores.map_partitions(pipeline.write_scores_on_dask)

    pipeline.write_scores_on_dask = pipeline.write_scores
    pipeline.write_scores = _write_scores

    if hasattr(pipeline, "post_processor"):
        # cannot use bob.pipelines.wrap here because the input is already a dask bag.
        pipeline.post_processor = bob.pipelines.DaskWrapper(
            pipeline.post_processor
        )

    return pipeline


def checkpoint_pipeline_simple(
    pipeline: PipelineSimple,
    base_dir: str,
    biometric_algorithm_dir: Optional[str] = None,
    hash_fn: Optional[Callable[[str], str]] = None,
    force: bool = False,
):
    """
    Given a :any:`PipelineSimple`, wraps their :attr:`transformer` and
    :attr:`biometric_algorithm` to be checkpointed.

    If an estimator of the pipeline is already checkpointed, it will not be
    wrapped again.

    Parameters
    ----------

    pipeline
       pipeline to be checkpointed

    base_dir
       Path to store transformed input data and possibly biometric references and scores

    biometric_algorithm_dir
       If set, it will checkpoint the biometric references and scores to this path.
       If not, `base_dir` will be used.
       This is useful when it's suitable to have the transformed data path, and biometric references and scores
       in different paths.

    hash_fn
       Pointer to a hash function. This hash function will map
       `sample.key` to a hash code and this hash code will be the
       relative directory where a single `sample` will be checkpointed.
       This is useful when is desireable file directories with more than
       a certain number of files.
    force
       Overwrite existing checkpoint files.
    """

    bio_ref_scores_dir = (
        base_dir if biometric_algorithm_dir is None else biometric_algorithm_dir
    )

    if isinstance(pipeline.transformer, Pipeline):
        for step, (name, transformer) in enumerate(pipeline.transformer.steps):
            if not is_instance_nested(
                transformer, "estimator", CheckpointWrapper
            ):
                pipeline.transformer.steps[step] = (
                    name,
                    bob.pipelines.wrap(
                        ["checkpoint"],
                        transformer,
                        features_dir=os.path.join(base_dir, name),
                        model_path=os.path.join(base_dir, name),
                        hash_fn=hash_fn,
                        force=force,
                    ),
                )
    else:  # The pipeline.transformer is a lone transformer
        if not is_instance_nested(
            pipeline.transformer, "estimator", CheckpointWrapper
        ):
            pipeline.transformer = bob.pipelines.wrap(
                ["checkpoint"],
                pipeline.transformer,
                features_dir=base_dir,
                model_path=base_dir,
                hash_fn=hash_fn,
                force=force,
            )

    pipeline.biometric_algorithm = BioAlgCheckpointWrapper(
        pipeline.biometric_algorithm,
        base_dir=bio_ref_scores_dir,
        hash_fn=hash_fn,
        force=force,
    )

    return pipeline


def is_biopipeline_checkpointed(pipeline: PipelineSimple) -> bool:
    """
    Check if :any:`PipelineSimple` is checkpointed

    Parameters
    ----------

    pipeline
       pipeline to check if checkpointed by a :any:`BioAlgCheckpointWrapper`.

    """

    # We have to check if biometric_algorithm is checkpointed
    return is_instance_nested(
        pipeline, "biometric_algorithm", BioAlgCheckpointWrapper
    )
