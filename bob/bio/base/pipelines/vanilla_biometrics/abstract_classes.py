from abc import ABCMeta, abstractmethod
from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import functools


class BioAlgorithm(metaclass=ABCMeta):
    """Describes a base biometric comparator for the Vanilla Biometrics Pipeline :ref:`_bob.bio.base.struct_bio_rec_sys`_.

    biometric model enrollement, via ``enroll()`` and scoring, with
    ``score()``.

    Parameters
    ----------

        allow_score_multiple_references: bool
          If true will call `self.score_multiple_biometric_references`, at scoring time, to compute scores in one shot with multiple probes.
          This optiization is useful when all probes needs to be compared with all biometric references AND
          your scoring function allows this broadcast computation.

    """

    def __init__(self, allow_score_multiple_references=False):
        self.allow_score_multiple_references = allow_score_multiple_references
        self.stacked_biometric_references = None

    def enroll_samples(self, biometric_references):
        """This method should implement the sub-pipeline 1 of the Vanilla Biometrics Pipeline :ref:`_vanilla-pipeline-1`.

        It handles the creation of biometric references

        Parameters
        ----------
            biometric_references : list
                A list of :py:class:`SampleSet` objects to be used for
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

        # Enroll
        return Sample(self.enroll(data), parent=sampleset)

    @abstractmethod
    def enroll(self, data):
        """
        It handles the creation of ONE biometric reference for the vanilla ppipeline

        Parameters
        ----------

            data:
                Data used for the creation of ONE BIOMETRIC REFERENCE

        """
        pass

    def score_samples(self, probe_features, biometric_references):
        """Scores a new sample against multiple (potential) references

        Parameters
        ----------

            probes : list
                A list of :py:class:`SampleSet` objects to be used for
                scoring the input references

            biometric_references : list
                A list of :py:class:`Sample` objects to be used for
                scoring the input probes, must have an ``id`` attribute that
                will be used to cross-reference which probes need to be scored.

        Returns
        -------

            scores : list
                For each sample in a probe, returns as many scores as there are
                samples in the probe, together with the probe's and the
                relevant reference's subject identifiers.

        """

        retval = []
        for p in probe_features:
            retval.append(self._score_sample_set(p, biometric_references))
        return retval

    def _score_sample_set(self, sampleset, biometric_references):
        """Given a sampleset for probing, compute the scores and retures a sample set with the scores
        """

        # Stacking the samples from a sampleset
        data = [s.data for s in sampleset.samples]

        # Compute scores for each sample inside of the sample set
        # TODO: In some cases we want to compute 1 score per sampleset (IJB-C)
        # We should add an agregator function here so we can properlly agregate samples from
        # a sampleset either after or before scoring.
        # To be honest, this should be the default behaviour
        retval = []

        def _write_sample(ref, probe, score):
            data = make_four_colums_score(ref.subject, probe.subject, probe.path, score)
            return Sample(data, parent=ref)

        for subprobe_id, (s, parent) in enumerate(zip(data, sampleset.samples)):
            # Creating one sample per comparison
            subprobe_scores = []

            if self.allow_score_multiple_references:
                # Multiple scoring
                if self.stacked_biometric_references is None:
                    self.stacked_biometric_references = [
                        ref.data for ref in biometric_references
                    ]
                scores = self.score_multiple_biometric_references(
                    self.stacked_biometric_references, s
                )

                # Wrapping the scores in samples
                for ref, score in zip(biometric_references, scores):
                    subprobe_scores.append(_write_sample(ref, sampleset, score[0]))
            else:

                for ref in [
                    r for r in biometric_references if r.key in sampleset.references
                ]:
                    score = self.score(ref.data, s)
                    subprobe_scores.append(_write_sample(ref, sampleset, score))

            # Creating one sampleset per probe
            subprobe = SampleSet(subprobe_scores, parent=sampleset)
            subprobe.subprobe_id = subprobe_id
            retval.append(subprobe)

        return retval

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

    @abstractmethod
    def score_multiple_biometric_references(self, biometric_references, data):
        """
        It handles the score computation of one probe and multiple biometric references
        This method is called is called if `allow_scoring_multiple_references` is set to true

        Parameters
        ----------

            biometric_references: list
                List of biometric references to be scored
            data:
                Data used for the creation of ONE BIOMETRIC REFERENCE

        """
        pass


class Database(metaclass=ABCMeta):
    """Base class for Vanilla Biometric pipeline
    """

    @abstractmethod
    def background_model_samples(self):
        """Returns :py:class:`Sample`'s to train a background model


        Returns
        -------
        samples : list
            List of samples for background model training.

        """
        pass

    @abstractmethod
    def references(self, group="dev"):
        """Returns :py:class:`Reference`'s to enroll biometric references


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
        """Returns :py:class:`Probe`'s to score biometric references


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


def make_four_colums_score(
    biometric_reference_subject, probe_subject, probe_path, score,
):
    data = "{0} {1} {2} {3}\n".format(
        biometric_reference_subject, probe_subject, probe_path, score,
    )
    return data


def create_score_delayed_sample(path, probe):
    """
    Write scores in the four columns format
    """

    with open(path, "w") as f:
        for score_line in probe.samples:
            f.write(score_line.data)

    def load():
        with open(path) as f:
            return f.read()

    return DelayedSample(load, parent=probe)
