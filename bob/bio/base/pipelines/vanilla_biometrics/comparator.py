#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @author: Andre Anjos <andre.anjos@idiap.ch>

from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import numpy


class Comparator(object):
    """Describes a base biometric comparator for the Vanilla Biometrics Pipeline :ref:`_bob.bio.base.struct_bio_rec_sys`_.

    biometric model enrollement, via ``enroll()`` and scoring, with
    ``score()``.

    """

    def __init__(self):
        pass

    def _stack_samples_2_ndarray(self, samplesets, stack_per_sampleset=False):
        """
        Stack a set of :py:class:`bob.pipelines.sample.sample.SampleSet`
        and convert them to :py:class:`numpy.ndarray`

        Parameters
        ----------

            samplesets: :py:class:`bob.pipelines.sample.sample.SampleSet`
                         Set of samples to be stackted

            stack_per_sampleset: bool
                If true will return a list of :py:class:`numpy.ndarray`, each one for a sample set

        """

        if stack_per_sampleset:
            # TODO: Make it more efficient
            all_data = []
            for sampleset in samplesets:
                all_data.append(
                    numpy.array([sample.data for sample in sampleset.samples])
                )
            return all_data
        else:
            return numpy.array(
                [
                    sample.data
                    for sampleset in samplesets
                    for sample in sampleset.samples
                ]
            )


    def enroll_samples(
        self, references, background_model=None, checkpoint=None, *args, **kwargs
    ):
        """This method should implement the sub-pipeline 1 of the Vanilla Biometrics Pipeline :ref:`_vanilla-pipeline-1`.

        It handles the creation of biometric references

        Parameters
        ----------
            references : list
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
        for k in references:
            # compute on-the-fly            
            data = [s.data for s in k.samples]
            retval.append(Sample(self.enroll(data), parent=k))

        return retval

    def write_biometric_reference(self, biometric_reference, filename):
        """Writes the enrolled model to the given file.
        In this base class implementation:

        - If the given model has a 'save' attribute, it calls ``model.save(bob.io.base.HDF5File(model_file), 'w')``.
          In this case, the given model_file might be either a file name or a :py:class:`bob.io.base.HDF5File`.
        - Otherwise, it uses :py:func:`bob.io.base.save` to do that.

        If you have a different format, please overwrite this function.

        **Parameters:**

        model : object
          A model as returned by the :py:meth:`enroll` function, which should be written.

        model_file : str or :py:class:`bob.io.base.HDF5File`
          The file open for writing, or the file name to write to.
        """
        import h5py

        with h5py.File(filename, "w") as f:
            f.create_dataset("biometric_reference", data=biometric_reference)

    def read_biometric_reference(self, filename):
        import h5py

        with h5py.File(filename, "r") as f:
            data = f["biometric_reference"].value
        return data

    def enroll(self, data,  **kwargs):
        """
        It handles the creation of ONE biometric reference for the vanilla ppipeline

        Parameters
        ----------

            data:
                Data used for the creation of ONE BIOMETRIC REFERENCE        

        """

        raise NotImplemented("Please, implement me")


    def score_samples(self, probes, references, background_model=None, *args, **kwargs):
        """Scores a new sample against multiple (potential) references

        Parameters
        ----------

            probes : list
                A list of :py:class:`SampleSet` objects to be used for
                scoring the input references

            references : list
                A list of :py:class:`Sample` objects to be used for
                scoring the input probes, must have an ``id`` attribute that
                will be used to cross-reference which probes need to be scored.

            background_model : 
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
            #data = numpy.vstack([s for s in p.samples])
            data = [s.data for s in p.samples]


            for subprobe_id, (s, parent) in enumerate(zip(data, p.samples)):
                # each sub-probe in the probe needs to be checked
                subprobe_scores = []
                for ref in [r for r in references if r.key in p.references]:
                    subprobe_scores.append(
                        Sample(self.score(ref.data, s), parent=ref)
                    )
                subprobe = SampleSet(subprobe_scores, parent=p)
                subprobe.subprobe_id = subprobe_id
                retval.append(subprobe)
        return retval


    def score(self, biometric_reference, data,  **kwargs):
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


import scipy.spatial.distance
from sklearn.utils.validation import check_array
class DistanceComparator(Comparator):

    def __init__(self,distance_function = scipy.spatial.distance.euclidean,factor=1):

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


    def score(self, model, probe,  **kwargs):
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
