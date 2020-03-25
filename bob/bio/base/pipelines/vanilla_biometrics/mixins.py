from bob.pipelines.mixins import CheckpointMixin
from bob.pipelines.sample import DelayedSample
import bob.io.base
import os
import functools
import dask
from .abstract_classes import save_scores_four_columns


class BioAlgCheckpointMixin(CheckpointMixin):
    """Mixing used to checkpoint Enrolled and Scoring samples.

    Examples
    --------

    >>> from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import BioAlgCheckpointMixin, Distance
    >>> class DistanceCheckpoint(BioAlgCheckpointMixin, Distance) pass:
    >>> biometric_algorithm = DistanceCheckpoint(features_dir="./")
    >>> biometric_algorithm.enroll(sample)

    It's possible to use it as with the :py:func:`bob.pipelines.mixins.mix_me_up`

    >>> from bob.pipelines.mixins import mix_me_up
    >>> biometric_algorithm = mix_me_up([BioAlgCheckpointMixin], Distance)(features_dir="./")
    >>> biometric_algorithm.enroll(sample)

    """

    def __init__(self, features_dir, **kwargs):
        super().__init__(features_dir=features_dir, **kwargs)
        self.biometric_reference_dir = os.path.join(
            features_dir, "biometric_references"
        )
        self.score_dir = os.path.join(features_dir, "scores")

    def save(self, sample, path):
        return bob.io.base.save(sample.data, path, create_directories=True)

    def _enroll_sample_set(self, sampleset):
        """
        Enroll a sample set with checkpointing
        """

        # Amending `models` directory
        path = os.path.join(
            self.biometric_reference_dir, str(sampleset.key) + self.extension
        )
        if path is None or not os.path.isfile(path):

            # Enrolling the sample
            enrolled_sample = super()._enroll_sample_set(sampleset)

            # saving the new sample
            self.save(enrolled_sample, path)

            # Dealaying it.
            # This seems inefficient, but it's crucial for large datasets
            delayed_enrolled_sample = DelayedSample(
                functools.partial(bob.io.base.load, path), enrolled_sample
            )

        else:
            # If sample already there, just load
            delayed_enrolled_sample = self.load(path)
            delayed_enrolled_sample.key = sampleset.key

        return delayed_enrolled_sample

    def _score_sample_set(self, sampleset, biometric_references):
        """Given a sampleset for probing, compute the scores and retures a sample set with the scores
        """
        # Computing score
        scored_sample_set = super()._score_sample_set(sampleset, biometric_references)

        for s in scored_sample_set:
            # Checkpointing score
            path = os.path.join(self.score_dir, str(s.path) + ".txt")
            bob.io.base.create_directories_safe(os.path.dirname(path))

            delayed_scored_sample = save_scores_four_columns(path, s)
            s.samples = [delayed_scored_sample]

        return scored_sample_set


class BioAlgDaskMixin:
    def enroll_samples(self, biometric_reference_features):
        biometric_references = biometric_reference_features.map_partitions(
            self.enroll_samples
        )
        return biometric_references

    def score_samples(self, probe_features, biometric_references):

        # TODO: Here, we are sending all computed biometric references to all
        # probes.  It would be more efficient if only the models related to each
        # probe are sent to the probing split.  An option would be to use caching
        # and allow the ``score`` function above to load the required data from
        # the disk, directly.  A second option would be to generate named delays
        # for each model and then associate them here.

        all_references = dask.delayed(list)(biometric_references)

        scores = probe_features.map_partitions(self.score_samples, all_references)
        return scores
