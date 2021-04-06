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
from .zt_norm import ZTNormPipeline, ZTNormDaskWrapper
from .legacy import BioAlgorithmLegacy
from bob.bio.base.transformers import (
    PreprocessorTransformer,
    ExtractorTransformer,
    AlgorithmTransformer,
)
from bob.pipelines.wrappers import SampleWrapper, CheckpointWrapper
from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
import logging
from bob.pipelines.utils import isinstance_nested
import gc
import time
from . import pickle_compress, uncompress_unpickle

logger = logging.getLogger(__name__)


class BioAlgorithmCheckpointWrapper(BioAlgorithm):
    """Wrapper used to checkpoint enrolled and Scoring samples.

    Parameters
    ----------
        biometric_algorithm: :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm`
           An implemented :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm`
    
        base_dir: str
           Path to store biometric references and scores
        
        extension: str
            File extension

        force: bool
          If True, will recompute scores and biometric references no matter if a file exists

    Examples
    --------

    >>> from bob.bio.base.pipelines.vanilla_biometrics import BioAlgorithmCheckpointWrapper, Distance    
    >>> biometric_algorithm = BioAlgorithmCheckpointWrapper(Distance(), base_dir="./")
    >>> biometric_algorithm.enroll(sample) # doctest: +SKIP

    """

    def __init__(
        self, biometric_algorithm, base_dir, group=None, force=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.base_dir = base_dir
        self.set_score_references_path(group)

        self.biometric_algorithm = biometric_algorithm
        self.force = force
        self._biometric_reference_extension = ".hdf5"
        self._score_extension = ".pickle.gz"

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
        return bob.io.base.save(sample.data, path, create_directories=True)

    def write_scores(self, samples, path):
        pickle_compress(path, samples)

    def _enroll_sample_set(self, sampleset):
        """
        Enroll a sample set with checkpointing
        """

        # Amending `models` directory
        path = os.path.join(
            self.biometric_reference_dir,
            str(sampleset.key) + self._biometric_reference_extension,
        )
        if self.force or not os.path.exists(path):

            enrolled_sample = self.biometric_algorithm._enroll_sample_set(sampleset)

            # saving the new sample
            self.write_biometric_reference(enrolled_sample, path)

        # This seems inefficient, but it's crucial for large datasets
        delayed_enrolled_sample = DelayedSample(
            functools.partial(bob.io.base.load, path), parent=sampleset
        )

        return delayed_enrolled_sample

    def _score_sample_set(
        self,
        sampleset,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):
        """Given a sampleset for probing, compute the scores and returns a sample set with the scores
        """

        def _load(path):
            return uncompress_unpickle(path)

        def _make_name(sampleset, biometric_references):
            # The score file name is composed by sampleset key and the
            # first 3 biometric_references
            reference_id = str(sampleset.reference_id)
            name = str(sampleset.key)
            suffix = "_".join([str(s.key) for s in biometric_references[0:3]])
            return os.path.join(reference_id, name + suffix)

        path = os.path.join(
            self.score_dir,
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


class BioAlgorithmDaskWrapper(BioAlgorithm):
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

    if isinstance(pipeline, ZTNormPipeline):
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

        pipeline.ztnorm_solver = ZTNormDaskWrapper(pipeline.ztnorm_solver)

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
    pipeline, base_dir, biometric_algorithm_dir=None, hash_fn=None
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

        # If they are legacy objects, we need to hook their load/save functions
        save_func = None
        load_func = None

        if not isinstance(estimator, SampleWrapper):
            raise ValueError(
                f"{estimator} needs to be the type `SampleWrapper` to be checkpointed"
            )

        if isinstance(estimator.estimator, PreprocessorTransformer):
            save_func = estimator.estimator.instance.write_data
            load_func = estimator.estimator.instance.read_data
        elif any(
            [
                isinstance(estimator.estimator, ExtractorTransformer),
                isinstance(estimator.estimator, AlgorithmTransformer),
            ]
        ):
            save_func = estimator.estimator.instance.write_feature
            load_func = estimator.estimator.instance.read_feature
            estimator.estimator.projector_file = os.path.join(
                bio_ref_scores_dir, "Projector.hdf5"
            )

        wraped_estimator = bob.pipelines.wrap(
            ["checkpoint"],
            estimator,
            features_dir=os.path.join(base_dir, name),
            load_func=load_func,
            save_func=save_func,
            hash_fn=hash_fn,
        )

        sk_pipeline.steps[i] = (name, wraped_estimator)

    if isinstance(pipeline.biometric_algorithm, BioAlgorithmLegacy):
        pipeline.biometric_algorithm.base_dir = bio_ref_scores_dir
    else:
        pipeline.biometric_algorithm = BioAlgorithmCheckpointWrapper(
            pipeline.biometric_algorithm, base_dir=bio_ref_scores_dir
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
