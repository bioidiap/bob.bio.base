#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import copy
import functools
import numpy
import os
import bob.io.base
from bob.pipelines.sample.sample import DelayedSample, SampleSet, Sample

"""Re-usable blocks for legacy bob.bio.base algorithms"""


class SampleLoader:
    """Adaptor for loading, preprocessing and feature extracting samples

    This adaptor class wraps around sample:

    .. code-block:: text

       [loading [-> preprocessing [-> extraction]]]

    The input sample object must obbey the following (minimal) API:

        * attribute ``id``: Contains an unique (string-fiable) identifier for
          processed samples
        * attribute ``data``: Contains the data for this sample

    Optional checkpointing is also implemented for each of the states,
    independently.  You may check-point just the preprocessing, feature
    extraction or both.


    Parameters
    ----------

    pipeline : :py:class:`list` of (:py:class:`str`, callable)
        A list of doubles in which the first entry are names of each processing
        step in the pipeline and second entry must be default-constructible
        :py:class:`bob.bio.base.preprocessor.Preprocessor` or
        :py:class:`bob.bio.base.preprocessor.Extractor` in any order.  Each
        of these objects must be a python type, that can be instantiated and
        used through its ``__call__()`` interface to process a single entry of
        a sample.  For python types that you may want to plug-in, but do not
        offer a default constructor that you like, pass the result of
        :py:func:`functools.partial` instead.

    """

    def __init__(self, pipeline):
        self.pipeline = copy.deepcopy(pipeline)

    def _handle_step(self, sset, func, checkpoint):
        """Handles a single step in the pipeline, with optional checkpointing

        Parameters
        ----------

        sset : SampleSet
            The original sample set to be processed (delayed or pre-loaded)

        func : callable
            The processing function to call for processing **each** sample in
            the set, if needs be

        checkpoint : str, None
            An optional string that may point to a directory that will be used
            for checkpointing the processing phase in question


        Returns
        -------

        r : SampleSet
            The prototype processed sample.  If no checkpointing required, this
            will be of type :py:class:`Sample`.  Otherwise, it will be a
            :py:class:`DelayedSample`

        """

        if checkpoint is not None:
            samples = []  # processed samples
            for s in sset.samples:
                # there can be a checkpoint for the data to be processed
                candidate = os.path.join(checkpoint, s.path + ".hdf5")
                if not os.path.exists(candidate):
                    # preprocessing is required, and checkpointing, do it now
                    data = func(s.data)

                    # notice this can be called in parallel w/o failing
                    bob.io.base.create_directories_safe(os.path.dirname(candidate))
                    # bob.bio.base standard interface for preprocessor
                    # has a read/write_data methods
                    writer = (
                        getattr(func, "write_data")
                        if hasattr(func, "write_data")
                        else getattr(func, "write_feature")
                    )
                    writer(data, candidate)

                # because we are checkpointing, we return a DelayedSample
                # instead of normal (preloaded) sample. This allows the next
                # phase to avoid loading it would it be unnecessary (e.g. next
                # phase is already check-pointed)
                reader = (
                    getattr(func, "read_data")
                    if hasattr(func, "read_data")
                    else getattr(func, "read_feature")
                )
                samples.append(
                    DelayedSample(functools.partial(reader, candidate), parent=s)
                )
        else:
            # if checkpointing is not required, load the data and preprocess it
            # as we would normally do
            samples = [Sample(func(s.data), parent=s) for s in sset.samples]

        r = SampleSet(samples, parent=sset)
        return r

    def _handle_sample(self, sset, pipeline):
        """Handles a single sampleset through a pipelien

        Parameters
        ----------

        sset : SampleSet
            The original sample set to be processed (delayed or pre-loaded)

        pipeline : :py:class:`list` of :py:class:`tuple`
            A list of tuples, each comprising of one processing function and
            one checkpoint directory (:py:class:`str` or ``None``, to avoid
            checkpointing that phase), respectively


        Returns
        -------

        r : Sample
            The processed sample

        """

        r = sset
        for func, checkpoint in pipeline:
            r = r if func is None else self._handle_step(r, func, checkpoint)
        return r

    def __call__(self, samples, checkpoints):
        """Applies the pipeline chaining with optional checkpointing

        Our implementation is optimized to minimize disk I/O to the most.  It
        yields :py:class:`DelayedSample`'s instead of :py:class:`Sample` if
        checkpointing is enabled.


        Parameters
        ----------

        samples : list
            List of :py:class:`SampleSet` to be treated by this pipeline

        checkpoints : dict
            A dictionary (with any number of entries) that may contain as many
            keys as those defined when you constructed this class with the
            pipeline tuple list.  Upon execution, the existance of an entry
            that defines checkpointing, this phase of the pipeline will be
            checkpointed.  Notice that you are in the control of checkpointing.
            If you miss an intermediary step, it will trigger this loader to
            load the relevant sample, even if the next phase is supposed to be
            checkpointed.  This strategy keeps the implementation as simple as
            possible.


        Returns
        -------

        samplesets : list
            Loaded samplesets, after optional preprocessing and extraction

        """

        pipe = [(v(), checkpoints.get(k)) for k, v in self.pipeline]
        return [self._handle_sample(k, pipe) for k in samples]


class VanillaBiometricsAlgoritm(object):
    """Describes a base biometric algorithm for the Vanilla Biometrics Pipeline :ref:`_bob.bio.base.struct_bio_rec_sys`_.

    The model can be fitted (optionally).  Otherwise, it can only execute
    biometric model enrollement, via ``enroll()`` and scoring, with
    ``score()``.

    """

    def __init__(self, performs_projection=False):
        self.performs_projection = performs_projection
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

    def fit(self, samplesets, checkpoint):
        """
        This method should implement the sub-pipeline 0 of the Vanilla Biometrics Pipeline :ref:`_vanilla-pipeline-0`.

        It represents the training of background models that an algorithm may need.

        Parameters
        ----------

            samplesets: :py:class:`bob.pipelines.sample.sample.SampleSet`
                         Set of samples used to train a background model


            checkpoint: str
                If provided, must the path leading to a location where this
                model should be saved at (complete path without extension) -
                currently, it needs to be provided because of existing
                serialization requirements (see bob/bob.io.base#106), but
                checkpointing will still work as expected.
         
        """
        raise NotImplemented("Please implement me")

    def enroll(
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

        def _project(k):
            return (
                self.project_one_sample(background_model, k.data)
                if self.performs_projection
                else k.data
            )

        retval = []
        for k in references:
            if checkpoint is not None:
                candidate = os.path.join(os.path.join(checkpoint, k.path + ".hdf5"))
                if not os.path.exists(candidate):
                    # create new checkpoint
                    bob.io.base.create_directories_safe(os.path.dirname(candidate))
                    data = numpy.vstack([_project(s) for s in k.samples])
                    enrolled = self.enroll_one_sample(data)
                    self.write_biometric_reference(enrolled, candidate)

                retval.append(
                    DelayedSample(
                        functools.partial(self.read_biometric_reference, candidate),
                        parent=k,
                    )
                )
            else:
                # compute on-the-fly
                data = _project(k)
                retval.append(Sample(model.enroll_one_sample(data), parent=k))

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

    def enroll_one_sample(self, data):
        """
        It handles the creation of ONE biometric reference for the vanilla ppipeline

        Parameters
        ----------

            data:
                Data used for the creation of ONE BIOMETRIC REFERENCE        

        """

        raise NotImplemented("Please, implement me")

    def project_one_sample(self, data):
        """
        If your method performs projection, it runs the projecttion

        Parameters
        ----------

            data:
                Data used for the projection of ONE BIOMETRIC REFERENCE        

        """

        raise NotImplemented("Please, implement me")

    def score(self, probes, references, background_model=None, *args, **kwargs):
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

        def _project(k):
            return (
                self.project_one_sample(background_model, k.data)
                if self.performs_projection
                else k.data
            )

        retval = []
        for p in probes:
            data = numpy.vstack([_project(s) for s in p.samples])

            for subprobe_id, (s, parent) in enumerate(zip(data, p.samples)):
                # each sub-probe in the probe needs to be checked
                subprobe_scores = []
                for ref in [r for r in references if r.id in p.references]:
                    subprobe_scores.append(
                        Sample(self.score_one_sample(ref.data, s), parent=ref)
                    )
                subprobe = SampleSet(subprobe_scores, parent=p)
                subprobe.subprobe_id = subprobe_id
                retval.append(subprobe)
        return retval

    def score_one_sample(self, biometric_reference, data):
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
