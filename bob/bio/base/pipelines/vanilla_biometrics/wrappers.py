from bob.pipelines import DelayedSample, SampleSet, Sample
import bob.io.base
import os
import dask
import functools
from .score_writers import FourColumnsScoreWriter
from .abstract_classes import BioAlgorithm
import bob.pipelines
import numpy as np
import h5py
import cloudpickle
from .zt_norm import ZTNormPipeline, ZTNormDaskWrapper
from .legacy import BioAlgorithmLegacy
from bob.bio.base.transformers import (
    PreprocessorTransformer,
    ExtractorTransformer,
    AlgorithmTransformer,
)
from bob.pipelines.wrappers import SampleWrapper
from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster


class BioAlgorithmCheckpointWrapper(BioAlgorithm):
    """Wrapper used to checkpoint enrolled and Scoring samples.

    Parameters
    ----------
        biometric_algorithm: :any:`BioAlgorithm`
           An implemented :any:`BioAlgorithm`
    
        base_dir: str
           Path to store biometric references and scores
        
        extension: str
            File extension

        force: bool
          If True, will recompute scores and biometric references no matter if a file exists

    Examples
    --------

    >>> from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import BioAlgCheckpointWrapper, Distance    
    >>> biometric_algorithm = BioAlgCheckpointWrapper(Distance(), base_dir="./")
    >>> biometric_algorithm.enroll(sample)

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
        self._score_extension = ".pkl"

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # cleaning parent
        with open(path, "wb") as f:
            f.write(cloudpickle.dumps(samples))

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
            return cloudpickle.loads(open(path, "rb").read())

            # with h5py.File(path) as hdf5:
            #    return hdf5_to_sample(hdf5)

        def _make_name(sampleset, biometric_references):
            # The score file name is composed by sampleset key and the
            # first 3 biometric_references
            subject = str(sampleset.subject)
            name = str(sampleset.key)
            suffix = "_".join([str(s.key) for s in biometric_references[0:3]])
            return os.path.join(subject, name + suffix)

        path = os.path.join(
            self.score_dir,
            _make_name(sampleset, biometric_references) + self._score_extension,
        )

        if self.force or not os.path.exists(path):

            # Computing score
            scored_sample_set = self.biometric_algorithm._score_sample_set(
                sampleset,
                biometric_references,
                allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
            )
            self.write_scores(scored_sample_set.samples, path)

            scored_sample_set = SampleSet(
                DelayedSample(functools.partial(_load, path), parent=sampleset),
                parent=sampleset,
            )
        else:
            samples = _load(path)
            scored_sample_set = SampleSet(samples, parent=sampleset)

        return scored_sample_set


class BioAlgorithmDaskWrapper(BioAlgorithm):
    """
    Wrap :any:`BioAlgorithm` to work with DASK
    """

    def __init__(self, biometric_algorithm, **kwargs):
        self.biometric_algorithm = biometric_algorithm

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
    Given a :any:`VanillaBiometrics`, wraps :any:`VanillaBiometrics.transformer` and
    :any:`VanillaBiometrics.biometric_algorithm` to be executed with dask

    Parameters
    ----------

    pipeline: :any:`VanillaBiometrics`
       Vanilla Biometrics based pipeline to be dasked

    npartitions: int
       Number of partitions for the initial :any:`dask.bag`

    partition_size: int
       Size of the partition for the initial :any:`dask.bag`
    """

    if isinstance(pipeline, ZTNormPipeline):
        # Dasking the first part of the pipelines
        pipeline.vanilla_biometrics_pipeline = dask_vanilla_biometrics(
            pipeline.vanilla_biometrics_pipeline, npartitions
        )
        pipeline.biometric_algorithm = pipeline.vanilla_biometrics_pipeline.biometric_algorithm
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

def dask_get_partition_size(cluster, n_objects):
    """
    Heuristics that gives you a number for dask.partition_size.
    The heuristics is pretty simple, given the max number of possible workers to be run
    in a queue (not the number of current workers running) and a total number objects to be processed do n_objects/n_max_workers:

    Parameters:
    -----------

        cluster:  :any:`bob.pipelines.distributed.SGEMultipleQueuesCluster`
            Cluster of the type :any:`bob.pipelines.distributed.SGEMultipleQueuesCluster`

        n_objects: int
            Number of objects to be processed

    """
    if not isinstance(cluster, SGEMultipleQueuesCluster):
        return None

    max_jobs = cluster.sge_job_spec["default"]["max_jobs"]
    return n_objects//max_jobs if n_objects>max_jobs else 1


def checkpoint_vanilla_biometrics(pipeline, base_dir):
    """
    Given a :any:`VanillaBiometrics`, wraps :any:`VanillaBiometrics.transformer` and
    :any:`VanillaBiometrics.biometric_algorithm` to be checkpointed

    Parameters
    ----------

    pipeline: :any:`VanillaBiometrics`
       Vanilla Biometrics based pipeline to be checkpointed

    base_dir: str
       Path to store biometric references and scores

    """

    sk_pipeline = pipeline.transformer
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

        wraped_estimator = bob.pipelines.wrap(
            ["checkpoint"],
            estimator,
            features_dir=os.path.join(base_dir, name),
            load_func=load_func,
            save_func=save_func,
        )

        sk_pipeline.steps[i] = (name, wraped_estimator)

    if isinstance(pipeline.biometric_algorithm, BioAlgorithmLegacy):
        pipeline.biometric_algorithm.base_dir = base_dir
    else:
        pipeline.biometric_algorithm = BioAlgorithmCheckpointWrapper(
            pipeline.biometric_algorithm, base_dir=base_dir
        )

    return pipeline
