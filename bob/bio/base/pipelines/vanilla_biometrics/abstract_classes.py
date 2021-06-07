#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from abc import ABCMeta, abstractmethod
from bob.pipelines.sample import Sample, SampleSet
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def average_scores(scores):
    """
    Given a :any:`numpy.ndarray` coming from multiple probes,
    average them
    """
    return np.mean(scores, axis=0)


class BioAlgorithm(metaclass=ABCMeta):
    """Describes a base biometric comparator for the Vanilla Biometrics Pipeline :ref:`bob.bio.base.biometric_algorithm`.

    biometric model enrollment, via ``enroll()`` and scoring, with
    ``score()``.

    Parameters
    ----------

        score_reduction_operation: ``collections.callable``
           Callable containing the score reduction function to be applied in the samples in a sampleset

    """

    def __init__(self, score_reduction_operation=average_scores, **kwargs):
        self.stacked_biometric_references = None
        self.score_reduction_operation = average_scores

    def clear_caches(self):
        """
        Clean all cached objects from BioAlgorithm
        """
        self.stacked_biometric_references = None

    def enroll_samples(self, biometric_references):
        """This method should implement the enrollment sub-pipeline of the Vanilla Biometrics Pipeline. TODO REF

        It handles the creation of biometric references

        Parameters
        ----------
            biometric_references : list
                A list of :any:`bob.pipelines.SampleSet` objects to be used for
                creating biometric references.  The sets must be identified
                with a unique id and a path, for eventual checkpointing.
        """

        retval = []
        for k in biometric_references:
            # compute on-the-fly
            retval.append(self._enroll_sample_set(k))

        return retval

    def _enroll_sample_set(self, sampleset):

        # Unpack the sampleset
        data = [s.data for s in sampleset.samples]

        valid_data = [d for d in data if d is not None]
        if len(data) != len(valid_data):
            logger.warning(
                f"Removed {len(data)-len(valid_data)} invalid enrollment samples."
            )
        if not valid_data:
            raise ValueError(
                f"None of the enrollment samples were valid for {sampleset}."
            )

        # Enroll
        return Sample(self.enroll(valid_data), parent=sampleset)

    @abstractmethod
    def enroll(self, data):
        """
        It handles the creation of ONE biometric reference for the vanilla pipeline

        Parameters
        ----------

            data:
                Data used for the creation of ONE BIOMETRIC REFERENCE

        """
        pass

    def score_samples(
        self,
        probe_features,
        biometric_references,
        allow_scoring_with_all_biometric_references=True,
    ):
        """Scores a new sample against multiple (potential) references

        Parameters
        ----------

            probes : list
                A list of :any:`bob.pipelines.SampleSet` objects to be used for
                scoring the input references

            biometric_references : list
                A list of :any:`bob.pipelines.Sample` objects to be used for
                scoring the input probes, must have an ``id`` attribute that
                will be used to cross-reference which probes need to be scored.

            allow_scoring_with_all_biometric_references: bool
                If true will call `self.score_multiple_biometric_references`, at scoring time, to compute scores in one shot with multiple probes.
                This optimization is useful when all probes needs to be compared with all biometric references AND
                your scoring function allows this broadcast computation.


        Returns
        -------

            scores : list
                For each sample in a probe, returns as many scores as there are
                samples in the probe, together with the probes and the
                relevant reference's subject identifiers.

        """

        retval = []
        for p in probe_features:
            retval.append(
                self._score_sample_set(
                    p,
                    biometric_references,
                    allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
                )
            )
        self.clear_caches()
        return retval

    def _score_sample_set(
        self,
        sampleset,
        biometric_references,
        allow_scoring_with_all_biometric_references,
    ):
        """Given one sampleset for probing, compute the scores and returns a sample set with the scores"""
        scores_biometric_references = []
        if allow_scoring_with_all_biometric_references:
            # Optimized scoring
            # This is useful when you scoring function can be compared with a
            # static batch of biometric references
            total_scores = []
            for probe_sample in sampleset:
                # Multiple scoring
                if self.stacked_biometric_references is None:
                    self.stacked_biometric_references = [
                        ref.data for ref in biometric_references
                    ]
                if probe_sample.data is None:
                    # Probe processing has failed. Mark invalid scores for FTA count
                    scores = [None] * len(self.stacked_biometric_references)
                else:
                    scores = self.score_multiple_biometric_references(
                        self.stacked_biometric_references, probe_sample.data
                    )
                total_scores.append(scores)

            # Reducing them
            total_scores = self.score_reduction_operation(
                np.array(total_scores, dtype=np.float)
            )

            # Wrapping the scores in samples
            for ref, score in zip(biometric_references, total_scores):
                scores_biometric_references.append(Sample(score, parent=ref))

        else:
            # Non optimizing scoring
            # There are some protocols where each probe has
            # to be scored with a specific list of biometric_references
            total_scores = []
            if self.stacked_biometric_references is None:
                self.stacked_biometric_references = dict()

            def cache_references(probe_refererences):
                """
                Stack references in a dictionary
                """
                for r in biometric_references:
                    if (
                        str(r.reference_id) in probe_refererences
                        and str(r.reference_id) not in self.stacked_biometric_references
                    ):
                        self.stacked_biometric_references[str(r.reference_id)] = r.data

            for probe_sample in sampleset:
                cache_references(sampleset.references)
                references = [
                    self.stacked_biometric_references[str(r.reference_id)]
                    for r in biometric_references
                    if str(r.reference_id) in sampleset.references
                ]

                if len(references) == 0:
                    raise ValueError(
                        f"The probe {sampleset} can't be compared with any biometric reference. "
                        "Something is probably wrong with your database interface."
                    )

                if probe_sample.data is None:
                    # Probe processing has failed
                    scores = [None] * len(self.stacked_biometric_references)
                else:
                    scores = self.score_multiple_biometric_references(
                        references, probe_sample.data
                    )

                total_scores.append(scores)

            total_scores = self.score_reduction_operation(
                np.array(total_scores, dtype=np.float)
            )

            for ref, score in zip(
                [
                    r
                    for r in biometric_references
                    if str(r.reference_id) in sampleset.references
                ],
                total_scores,
            ):

                scores_biometric_references.append(Sample(score, parent=ref))

        return SampleSet(scores_biometric_references, parent=sampleset)

    @abstractmethod
    def score(self, biometric_reference, data):
        """It handles the score computation for one sample

        Parameters
        ----------

            biometric_reference : list
                Biometric reference to be compared

            data : list
                Data to be compared

        Returns
        -------

            scores : list
                For each sample in a probe, returns as many scores as there are
                samples in the probe, together with the probe's and the
                relevant reference's subject identifiers.

        """
        pass

    def score_multiple_biometric_references(self, biometric_references, data):
        """Score one probe against multiple biometric references (models).
        This method is called if `allow_scoring_multiple_references` is set to true.
        You may want to override this method to improve the performance of computations.

        Parameters
        ----------
        biometric_references : list
            List of biometric references (models) to be scored
            [description]
        data
            Data used for the creation of ONE biometric probe.

        Returns
        -------
        list
            A list of scores for the comparison of the probe against multiple models.
        """
        return [self.score(model, data) for model in biometric_references]


class Database(metaclass=ABCMeta):
    """Base class for Vanilla Biometric pipeline"""

    def __init__(
        self,
        name,
        protocol,
        allow_scoring_with_all_biometric_references=False,
        annotation_type=None,
        fixed_positions=None,
        memory_demanding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.protocol = protocol
        self.allow_scoring_with_all_biometric_references = (
            allow_scoring_with_all_biometric_references
        )
        self.annotation_type = annotation_type
        self.fixed_positions = fixed_positions
        self.memory_demanding = memory_demanding

    @abstractmethod
    def background_model_samples(self):
        """Returns :any:`bob.pipelines.Sample`'s to train a background model


        Returns
        -------
        samples : list
            List of samples for background model training.

        """
        pass

    @abstractmethod
    def references(self, group="dev"):
        """Returns references to enroll biometric references


        Parameters
        ----------
        group : :py:class:`str`, optional
            Limits samples to this group


        Returns
        -------
        references : list
            List of samples for the creation of biometric references.

        """
        pass

    @abstractmethod
    def probes(self, group):
        """Returns probes to score biometric references


        Parameters
        ----------
        group : str
            Limits samples to this group


        Returns
        -------
        probes : list
            List of samples for the creation of biometric probes.

        """
        pass

    @abstractmethod
    def all_samples(self, groups=None):
        """Returns all the samples of the dataset

        Parameters
        ----------
        groups: list or `None`
            List of groups to consider (like 'dev' or 'eval'). If `None`, will
            return samples from all the groups.

        Returns
        -------
        samples: list
            List of all the samples of the dataset.
        """
        pass

    @abstractmethod
    def groups(self):
        pass

    @abstractmethod
    def protocols(self):
        pass

    def reference_ids(self, group):
        return [s.reference_id for s in self.references(group=group)]


class ScoreWriter(metaclass=ABCMeta):
    """
    Defines base methods to read, write scores and concatenate scores
    for :any:`bob.bio.base.pipelines.vanilla_biometrics.BioAlgorithm`
    """

    def __init__(self, path, extension=".txt"):
        self.path = path
        self.extension = extension

    @abstractmethod
    def write(self, sampleset, path):
        pass

    def post_process(self, score_paths, filename):
        def _post_process(score_paths, filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                for path in score_paths:
                    with open(path) as f2:
                        f.writelines(f2.readlines())
            return filename

        import dask.bag
        import dask

        if isinstance(score_paths, dask.bag.Bag):
            all_paths = dask.delayed(list)(score_paths)
            return dask.delayed(_post_process)(all_paths, filename)
        return _post_process(score_paths, filename)
