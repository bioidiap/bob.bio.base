from bob.pipelines import (
    DelayedSample,
    SampleSet,
    Sample,
    DelayedSampleSet,
    DelayedSampleSetCached,
)
import bob.io.base
import os
import dask
import functools
from .score_writers import FourColumnsScoreWriter
from .abstract_classes import BioAlgorithm
import bob.pipelines
import numpy as np
import h5py

# from .zt_norm import ZTNormPipeline, ZTNormDaskWrapper
from .score_post_processor import (
    ScoreNormalizationPipeline,
    dask_score_normalization_pipeline,
)
from .legacy import BioAlgorithmLegacy
from bob.bio.base.transformers import (
    PreprocessorTransformer,
    ExtractorTransformer,
    AlgorithmTransformer,
)
from bob.pipelines.wrappers import (
    SampleWrapper,
    CheckpointWrapper,
    get_bob_tags,
    BaseWrapper,
)
from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
import logging
from bob.pipelines.utils import isinstance_nested
import gc
import time
from . import pickle_compress, uncompress_unpickle

logger = logging.getLogger(__name__)


def default_save(data, path):
    return bob.io.base.save(data, path, create_directories=True)

def default_load(path):
    return bob.io.base.load(path)

def get_vanilla_biometrics_tags(estimator=None, force_tags=None):
    bob_tags = get_bob_tags(estimator=estimator, force_tags=force_tags)
    default_tags = {
        "bob_enrolled_extension": ".h5",
        "bob_enrolled_save_fn": default_save,
        "bob_enrolled_load_fn": default_load,
    }
    force_tags = force_tags or {}
    estimator_tags = estimator._get_tags() if estimator is not None else {}
    return {**bob_tags, **default_tags, **estimator_tags, **force_tags}

class BioAlgorithmCheckpointWrapper(BioAlgorithm, BaseWrapper):
    """Wrapper used to checkpoint enrolled and Scoring samples.

    Parameters
    ----------
    biometric_algorithm: :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm`
       An implemented :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm`

    base_dir: str
       Path to store biometric references and scores

    extension: str
       Default extension of the enrolled references files.
       If None, will use the ``bob_checkpoint_extension`` tag in the estimator, or
       default to ``.h5``.

    save_func : callable
       Pointer to a customized function that saves an enrolled reference to the disk.
       If None, will use the ``bob_enrolled_save_fn`` tag in the estimator, or default
       to ``bob.io.base.save``.

    load_func: callable
       Pointer to a customized function that loads an enrolled reference from disk.
       If None, will use the ``bob_enrolled_load_fn`` tag in the estimator, or default
       to ``bob.io.base.load``.

    force: bool
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

    >>> from bob.bio.base.pipelines.vanilla_biometrics import BioAlgorithmCheckpointWrapper, Distance
    >>> biometric_algorithm = BioAlgorithmCheckpointWrapper(Distance(), base_dir="./")
    >>> biometric_algorithm.enroll(sample) # doctest: +SKIP

    """

    def __init__(
        self,
        biometric_algorithm,
        base_dir,
        extension=None,
        save_func=None,
        load_func=None,
        group=None,
        force=False,
        hash_fn=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.base_dir = base_dir
        self.set_score_references_path(group)
        self.group = group
        self.biometric_algorithm = biometric_algorithm
        self.force = force
        self.hash_fn = hash_fn
        bob_tags = get_vanilla_biometrics_tags(self.biometric_algorithm)
        self.extension = extension or bob_tags["bob_enrolled_extension"]
        self.save_func = save_func or bob_tags["bob_enrolled_save_fn"]
        self.load_func = load_func or bob_tags["bob_enrolled_load_fn"]

        self._score_extension = ".pickle.gz"
        self._biometric_reference_extension = self.extension

    def clear_caches(self):
        self.biometric_algorithm.clear_caches()

    def set_score_references_path(self, group):
        if group is None:
            self.biometric_reference_dir = os.path.join(
                self.base_dir, "biometric_references"
            )
            self.score_dir = os.path.join(self.base_dir, "scores")
        else:
            self.biometric_reference_dir = os.path.join(
                self.base_dir, group, "biometric_references"
            )
            self.score_dir = os.path.join(self.base_dir, group, "scores")

    def enroll(self, enroll_features):
        return self.biometric_algorithm.enroll(enroll_features)

    def score(self, biometric_reference, data):
        return self.biometric_algorithm.score(biometric_reference, data)

    def score_multiple_biometric_references(self, biometric_references, data):
        return self.biometric_algorithm.score_multiple_biometric_references(
            biometric_references, data
        )

    def write_biometric_reference(self, sample, path):
        return self.save_func(sample.data, path)

    def write_scores(self, samples, path):
        pickle_compress(path, samples)

    def _enroll_sample_set(self, sampleset):
        """
        Enroll a sample set with checkpointing
        """

        # Amending `models` directory
        hash_dir_name = (
            self.hash_fn(str(sampleset.key)) if self.hash_fn is not None else ""
        )

        path = os.path.join(
            self.biometric_reference_dir,
            hash_dir_name,
            str(sampleset.key) + self._biometric_reference_extension,
        )

        if self.force or not os.path.exists(path):

            enrolled_sample = self.biometric_algorithm._enroll_sample_set(sampleset)

            # saving the new sample
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.write_biometric_reference(enrolled_sample, path)

        # This seems inefficient, but it's crucial for large datasets
        delayed_enrolled_sample = DelayedSample(
            functools.partial(self.load_func, path), parent=sampleset
        )

        return delayed_enrolled_sample

    def _score_sample_set(
        self,
        sampleset,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):
        """Given a sampleset for probing, compute the scores and returns a sample set with the scores"""

        def _load(path):
            return uncompress_unpickle(path)

        def _make_name(sampleset, biometric_references):
            # The score file name is composed by sampleset key and the
            # first 3 biometric_references
            reference_id = str(sampleset.reference_id)
            name = str(sampleset.key)
            suffix = "_".join([str(s.key) for s in biometric_references[0:3]])
            return os.path.join(reference_id, name + suffix)

        # Amending `models` directory
        hash_dir_name = (
            self.hash_fn(str(sampleset.key)) if self.hash_fn is not None else ""
        )

        path = os.path.join(
            self.score_dir,
            hash_dir_name,
            _make_name(sampleset, biometric_references) + self._score_extension,
        )

        parent = sampleset
        if self.force or not os.path.exists(path):

            # Computing score
            scored_sample_set = self.biometric_algorithm._score_sample_set(
                sampleset,
                biometric_references,
                allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
            )
            self.write_scores(scored_sample_set.samples, path)
            parent = scored_sample_set

        scored_sample_set = DelayedSampleSetCached(
            functools.partial(_load, path), parent=parent
        )

        return scored_sample_set


class BioAlgorithmDaskWrapper(BioAlgorithm, BaseWrapper):
    """
    Wrap :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm` to work with DASK
    """

    def __init__(self, biometric_algorithm, **kwargs):
        self.biometric_algorithm = biometric_algorithm

    def clear_caches(self):
        self.biometric_algorithm.clear_caches()

    def enroll_samples(self, biometric_reference_features):

        biometric_references = biometric_reference_features.map_partitions(
            self.biometric_algorithm.enroll_samples
        )

        return biometric_references

    def score_samples(
        self,
        probe_features,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        # TODO: Here, we are sending all computed biometric references to all
        # probes.  It would be more efficient if only the models related to each
        # probe are sent to the probing split.  An option would be to use caching
        # and allow the ``score`` function above to load the required data from
        # the disk, directly.  A second option would be to generate named delays
        # for each model and then associate them here.

        all_references = dask.delayed(list)(biometric_references)
        scores = probe_features.map_partitions(
            self.biometric_algorithm.score_samples,
            all_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )
        return scores

    def enroll(self, data):
        return self.biometric_algorithm.enroll(data)

    def score(self, biometric_reference, data):
        return self.biometric_algorithm.score(biometric_reference, data)

    def score_multiple_biometric_references(self, biometric_references, data):
        return self.biometric_algorithm.score_multiple_biometric_references(
            biometric_references, data
        )

    def set_score_references_path(self, group):
        self.biometric_algorithm.set_score_references_path(group)


def dask_vanilla_biometrics(pipeline, npartitions=None, partition_size=None):
    """
    Given a :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline`, wraps :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline` and
    :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm` to be executed with dask

    Parameters
    ----------

    pipeline: :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline`
       Vanilla Biometrics based pipeline to be dasked

    npartitions: int
       Number of partitions for the initial `dask.bag`

    partition_size: int
       Size of the partition for the initial `dask.bag`
    """

    if isinstance(pipeline, ScoreNormalizationPipeline):
        # Dasking the first part of the pipelines
        pipeline.vanilla_biometrics_pipeline = dask_vanilla_biometrics(
            pipeline.vanilla_biometrics_pipeline,
            npartitions=npartitions,
            partition_size=partition_size,
        )
        pipeline.biometric_algorithm = (
            pipeline.vanilla_biometrics_pipeline.biometric_algorithm
        )
        pipeline.transformer = pipeline.vanilla_biometrics_pipeline.transformer

        pipeline = dask_score_normalization_pipeline(pipeline)

    else:

        if partition_size is None:
            pipeline.transformer = bob.pipelines.wrap(
                ["dask"], pipeline.transformer, npartitions=npartitions
            )
        else:
            pipeline.transformer = bob.pipelines.wrap(
                ["dask"], pipeline.transformer, partition_size=partition_size
            )
        pipeline.biometric_algorithm = BioAlgorithmDaskWrapper(
            pipeline.biometric_algorithm
        )

        def _write_scores(scores):
            return scores.map_partitions(pipeline.write_scores_on_dask)

        pipeline.write_scores_on_dask = pipeline.write_scores
        pipeline.write_scores = _write_scores

    return pipeline


def checkpoint_vanilla_biometrics(
    pipeline, base_dir, biometric_algorithm_dir=None, hash_fn=None, force=False
):
    """
    Given a :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline`, wraps :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline` and
    :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm` to be checkpointed

    Parameters
    ----------

    pipeline: :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline`
       Vanilla Biometrics based pipeline to be checkpointed

    base_dir: str
       Path to store transformed input data and possibly biometric references and scores

    biometric_algorithm_dir: str
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
    """

    sk_pipeline = pipeline.transformer

    bio_ref_scores_dir = (
        base_dir if biometric_algorithm_dir is None else biometric_algorithm_dir
    )

    for i, name, estimator in sk_pipeline._iter():

        wrapped_estimator = bob.pipelines.wrap(
            ["checkpoint"],
            estimator,
            features_dir=os.path.join(base_dir, name),
            hash_fn=hash_fn,
            force=force,
        )

        sk_pipeline.steps[i] = (name, wrapped_estimator)

    if isinstance(pipeline.biometric_algorithm, BioAlgorithmLegacy):
        pipeline.biometric_algorithm.base_dir = bio_ref_scores_dir

        # Here we need to check if the LAST transformer is
        # 1. is instance of CheckpointWrapper
        # 2. Its estimator is instance of AlgorithmTransformer
        if (
            isinstance(pipeline.transformer[-1], CheckpointWrapper)
            and hasattr(pipeline.transformer[-1].estimator, "estimator")
            and isinstance(
                pipeline.transformer[-1].estimator.estimator, AlgorithmTransformer
            )
        ):

            pipeline.transformer[
                -1
            ].estimator.estimator.projector_file = bio_ref_scores_dir

    else:
        pipeline.biometric_algorithm = BioAlgorithmCheckpointWrapper(
            pipeline.biometric_algorithm,
            base_dir=bio_ref_scores_dir,
            hash_fn=hash_fn,
            force=force,
        )

    return pipeline


def is_checkpointed(pipeline):
    """
    Check if :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline` is checkpointed


    Parameters
    ----------

    pipeline: :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline`
       Vanilla Biometrics based pipeline to be checkpointed

    """

    # We have to check if is BioAlgorithmCheckpointWrapper OR
    # If it BioAlgorithmLegacy and the transformer of BioAlgorithmLegacy is also checkpointable
    return isinstance_nested(
        pipeline, "biometric_algorithm", BioAlgorithmCheckpointWrapper
    ) or (
        isinstance_nested(pipeline, "biometric_algorithm", BioAlgorithmLegacy)
        and isinstance_nested(pipeline, "transformer", CheckpointWrapper)
    )
