from bob.pipelines import DelayedSample
import bob.io.base
import os
import dask
import functools
from .score_writers import FourColumnsScoreWriter
from .abstract_classes import BioAlgorithm

import bob.pipelines as mario


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

        score_writer: :any:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classe.ScoreWriter`
            Format to write scores. Default to :any:`FourColumnsScoreWriter`

        force: bool
          If True, will recompute scores and biometric references no matter if a file exists

    Examples
    --------

    >>> from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import BioAlgCheckpointWrapper, Distance    
    >>> biometric_algorithm = BioAlgCheckpointWrapper(Distance(), base_dir="./")
    >>> biometric_algorithm.enroll(sample)

    """

    def __init__(
        self,
        biometric_algorithm,
        base_dir,
        score_writer=FourColumnsScoreWriter(),
        force=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.biometric_reference_dir = os.path.join(base_dir, "biometric_references")
        self.score_dir = os.path.join(base_dir, "scores")
        self.biometric_algorithm = biometric_algorithm
        self.force = force
        self._biometric_reference_extension = ".hdf5"
        self.score_writer = score_writer

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

        # TODO: WE CAN'T REUSE THE ALREADY WRITTEN SCORE FILE FOR LOADING
        #       UNLESS WE SAVE THE PICKLED THE SAMPLESET WITH THE SCORES

        path = os.path.join(self.score_dir, str(sampleset.key))

        # Computing score
        scored_sample_set = self.biometric_algorithm._score_sample_set(
            sampleset,
            biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        scored_sample_set = self.score_writer.write(scored_sample_set, path)

        return scored_sample_set


class BioAlgorithmDaskWrapper(BioAlgorithm):
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


def dask_vanilla_biometrics(vanila_biometrics_pipeline, npartitions=None):
    """
    Given a :any:`VanillaBiometrics`, wraps :any:`VanillaBiometrics.transformer` and
    :any:`VanillaBiometrics.biometric_algorithm` to be executed with dask

    Parameters
    ----------

    vanila_biometrics_pipeline: :any:`VanillaBiometrics`
       Vanilla Biometrics based pipeline to be dasked

    npartitions: int
       Number of partitions for the initial :any:`dask.bag`
    """

    vanila_biometrics_pipeline.transformer = mario.wrap(
        ["dask"], vanila_biometrics_pipeline.transformer, npartitions=npartitions
    )
    vanila_biometrics_pipeline.biometric_algorithm = BioAlgorithmDaskWrapper(
        vanila_biometrics_pipeline.biometric_algorithm
    )

    return vanila_biometrics_pipeline
