from abc import ABCMeta, abstractmethod
from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import functools


class BioAlgorithm(metaclass=ABCMeta):
    """Describes a base biometric comparator for the Vanilla Biometrics Pipeline :ref:`_bob.bio.base.struct_bio_rec_sys`_.

    biometric model enrollement, via ``enroll()`` and scoring, with
    ``score()``.

    """

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

        for subprobe_id, (s, parent) in enumerate(zip(data, sampleset.samples)):
            # Creating one sample per comparison
            subprobe_scores = []
            for ref in [
                r for r in biometric_references if r.key in sampleset.references
            ]:
                score = self.score(ref.data, s)
                data = make_score_line(ref.subject, sampleset.subject, sampleset.path, score)
                subprobe_scores.append(Sample(data, parent=ref))

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


def make_score_line(
    biometric_reference_subject, probe_subject, probe_path, score,
):
    data = "{0} {1} {2} {3}\n".format(
        biometric_reference_subject,
        probe_subject,
        probe_path,
        score,
    )
    return data


def save_scores_four_columns(path, probe):
    """
    Write scores in the four columns format
    """

    with open(path, "w") as f:
        for biometric_reference in probe.samples:
            line = make_score_line(
                biometric_reference.subject,
                probe.subject,
                probe.path,
                biometric_reference.data,
            )
            f.write(line)

    def load():
        with open(path) as f:
            return f.read()

    return DelayedSample(load, parent=probe)
