#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @author: Andre Anjos <andre.anjos@idiap.ch>

from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import numpy
import bob.io.base
import os
import functools


class BiometricAlgorithm(object):
    """Describes a base biometric comparator for the Vanilla Biometrics Pipeline :ref:`_bob.bio.base.struct_bio_rec_sys`_.

    biometric model enrollement, via ``enroll()`` and scoring, with
    ``score()``.

    """

    def __init__(self):
        pass

    def _enroll_samples(
        self, biometric_references, extractor=None, checkpoint=None, *args, **kwargs
    ):
        """This method should implement the sub-pipeline 1 of the Vanilla Biometrics Pipeline :ref:`_vanilla-pipeline-1`.

        It handles the creation of biometric references

        Parameters
        ----------
            biometric_references : list
                A list of :py:class:`SampleSet` objects to be used for
                creating biometric references.  The sets must be identified
                with a unique id and a path, for eventual checkpointing.

            background_model : 
                Object containing the background model

            checkpoint : str, None
                If passed and not ``None``, then it is considered to be the
                path of a directory containing possible cached values for each
                of the references in this experiment.  If that is the case, the
                values are loaded from there and not recomputed.

            *args, **kwargs :
                Extra parameters that can be used to hook-up processing graph
                dependencies, but are currently ignored

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


    def enroll(self, data, extractor=None, **kwargs):
        """
        It handles the creation of ONE biometric reference for the vanilla ppipeline

        Parameters
        ----------

            data:
                Data used for the creation of ONE BIOMETRIC REFERENCE        

        """

        raise NotImplemented("Please, implement me")


    def _score_samples(self, probes, biometric_references, extractor=None, *args, **kwargs):
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

            extractor : 
                Path pointing to stored model on disk

            *args, **kwargs :
                Extra parameters that can be used to hook-up processing graph
                dependencies, but are currently ignored


        Returns
        -------

            scores : list
                For each sample in a probe, returns as many scores as there are
                samples in the probe, together with the probe's and the
                relevant reference's subject identifiers.

        """

        retval = []
        for p in probes:
            retval.append(self._score_sample_set(p, biometric_references, extractor))
        return retval


    def _score_sample_set(self, sampleset, biometric_references, extractor):
        """Given a sampleset for probing, compute the scores and retures a sample set with the scores
        """

        # Stacking the samples from a sampleset        
        data = [s.data for s in sampleset.samples]

        # Compute scores for each sample inside of the sample set
        # TODO: In some cases we want to compute 1 score per sampleset (IJB-C)
        # We should add an agregator function here so we can properlly agregate samples from 
        # a sampleset either after or before scoring.
        # To be honest, this should be the default behaviour
        for subprobe_id, (s, parent) in enumerate(zip(data, sampleset.samples)):
            # Creating one sample per comparison
            subprobe_scores = []

            for ref in [r for r in biometric_references if r.key in sampleset.references]:
                subprobe_scores.append(
                    Sample(self.score(ref.data, s, extractor), parent=ref)
                )
            # Creating one sampleset per probe
            subprobe = SampleSet(subprobe_scores, parent=sampleset)
            subprobe.subprobe_id = subprobe_id

        return subprobe


    def score(self, biometric_reference, data, extractor=None,  **kwargs):
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
        raise NotImplemented("Please, implement me")


from bob.pipelines.mixins import CheckpointMixin
class BiometricAlgorithmCheckpointMixin(CheckpointMixin):
    """Mixing used to checkpoint Enrolled and Scoring samples.

    Examples
    --------

    >>> from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import BiometricAlgorithmCheckpointMixin, Distance
    >>> class DistanceCheckpoint(BiometricAlgorithmCheckpointMixin, Distance) pass:
    >>> biometric_algorithm = DistanceCheckpoint(features_dir="./")
    >>> biometric_algorithm.enroll(sample)

    It's possible to use it as with the :py:func:`bob.pipelines.mixins.mix_me_up` 

    >>> from bob.pipelines.mixins import mix_me_up
    >>> biometric_algorithm = mix_me_up([BiometricAlgorithmCheckpointMixin], Distance)(features_dir="./")
    >>> biometric_algorithm.enroll(sample)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.biometric_reference_dir = os.path.join(self.features_dir, "biometric_references")
        self.score_dir = os.path.join(self.features_dir, "scores")


    def save(self, sample, path):
        return bob.io.base.save(sample.data, path, create_directories=True)


    def _enroll_sample_set(self, sampleset):
        """
        Enroll a sample set with checkpointing
        """

        # Amending `models` directory
        path = os.path.join(self.biometric_reference_dir, str(sampleset.key) + self.extension)
        if path is None or not os.path.isfile(path):

            # Enrolling the sample
            enrolled_sample = super()._enroll_sample_set(sampleset)

            # saving the new sample
            self.save(enrolled_sample, path)

            # Dealaying it.
            # This seems inefficient, but it's crucial for large datasets
            delayed_enrolled_sample = DelayedSample(functools.partial(bob.io.base.load, path), enrolled_sample)

        else:
            # If sample already there, just load
            delayed_enrolled_sample = self.load(path)
            delayed_enrolled_sample.key = sampleset.key


        return delayed_enrolled_sample


    def _score_sample_set(self, sampleset, biometric_references, extractor):
        """Given a sampleset for probing, compute the scores and retures a sample set with the scores
        """
        # Computing score
        scored_sample_set = super()._score_sample_set(sampleset, biometric_references, extractor)

        # Checkpointing score
        path = os.path.join(self.score_dir, str(sampleset.key) + ".txt")
        bob.io.base.create_directories_safe(os.path.dirname(path))

        delayed_scored_sample = save_scores_four_columns(path, scored_sample_set)
        scored_sample_set.samples = [delayed_scored_sample]
        return scored_sample_set


import scipy.spatial.distance
from sklearn.utils.validation import check_array
class Distance(BiometricAlgorithm):

    def __init__(self,distance_function = scipy.spatial.distance.euclidean,factor=-1):

        self.distance_function = distance_function
        self.factor = factor


    def enroll(self, enroll_features,  **kwargs):
        """enroll(enroll_features) -> model

        Enrolls the model by storing all given input vectors.

        Parameters:
        -----------

        ``enroll_features`` : [:py:class:`numpy.ndarray`]
          The list of projected features to enroll the model from.

        Returns:
        --------

        ``model`` : 2D :py:class:`numpy.ndarray`
	      The enrolled model.
        """

        enroll_features = check_array(enroll_features, allow_nd=True)

        return numpy.mean(enroll_features, axis=0)


    def score(self, model, probe, extractor=None, **kwargs):
        """score(model, probe) -> float

        Computes the distance of the model to the probe using the distance function specified in the constructor.

        Parameters:
	    -----------

        ``model`` : 2D :py:class:`numpy.ndarray`
          The model storing all enrollment features

        ``probe`` : :py:class:`numpy.ndarray`
          The probe feature vector

        Returns:
        --------

        ``score`` : float
          A similarity value between ``model`` and ``probe``
        """

        probe = probe.flatten()
        # return the negative distance (as a similarity measure)
        return self.factor * self.distance_function(model, probe)


def save_scores_four_columns(path, probe):
    """
    Write scores in the four columns format
    """
    
    with open(path, "w") as f:
        for biometric_reference in probe.samples:
            line = "{0} {1} {2} {3}\n".format(biometric_reference.key, probe.key, probe.path, biometric_reference.data)
            f.write(line)

    return DelayedSample(functools.partial(open, path))

