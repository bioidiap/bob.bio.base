#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Re-usable blocks for legacy bob.bio.base algorithms"""

import os
import copy
import functools

import bob.io.base
from bob.pipelines.sample.sample import DelayedSample, SampleSet, Sample
import numpy
import logging
import dask

import sys
import pickle
logger = logging.getLogger("bob.bio.base")


class DatabaseConnector:
    """Wraps a bob.bio.base database and generates conforming samples

    This connector allows wrapping generic bob.bio.base datasets and generate
    samples that conform to the specifications of biometric pipelines defined
    in this package.


    Parameters
    ----------

    database : object
        An instantiated version of a bob.bio.base.Database object

    protocol : str
        The name of the protocol to generate samples from.
        To be plugged at :py:method:`bob.db.base.Database.objects`.

    """

    def __init__(self, database, protocol):
        self.database = database
        self.protocol = protocol
        self.directory = database.original_directory
        self.extension = database.original_extension

    def background_model_samples(self):
        """Returns :py:class:`Sample`'s to train a background model (group
        ``world``).


        Returns
        -------

            samples : list
                List of samples conforming the pipeline API for background
                model training.  See, e.g., :py:func:`.pipelines.first`.

        """

        # TODO: This should be organized by client
        retval = []

        objects = self.database.objects(protocol=self.protocol, groups="world")

        return [
            SampleSet(
                [
                    DelayedSample(
                        load=functools.partial(
                            k.load,
                            self.database.original_directory,
                            self.database.original_extension,
                        ),
                        id=k.id,
                        path=k.path,
                    )
                ]
            )
            for k in objects
        ]

    def references(self, group="dev"):
        """Returns :py:class:`Reference`'s to enroll biometric references


        Parameters
        ----------

            group : :py:class:`str`, optional
                A ``group`` to be plugged at
                :py:meth:`bob.db.base.Database.objects`


        Returns
        -------

            references : list
                List of samples conforming the pipeline API for the creation of
                biometric references.  See, e.g., :py:func:`.pipelines.first`.

        """

        retval = []

        for m in self.database.model_ids(protocol=self.protocol, groups=group):

            objects = self.database.objects(
                protocol=self.protocol, groups=group, model_ids=(m,), purposes="enroll"
            )

            retval.append(
                SampleSet(
                    [
                        DelayedSample(
                            load=functools.partial(
                                k.load,
                                self.database.original_directory,
                                self.database.original_extension,
                            ),
                            id=k.id,
                            path=k.path,
                        )
                        for k in objects
                    ],
                    id=m,
                    path=str(m),
                    subject=objects[0].client_id,
                )
            )

        return retval

    def probes(self, group):
        """Returns :py:class:`Probe`'s to score biometric references


        Parameters
        ----------

            group : str
                A ``group`` to be plugged at
                :py:meth:`bob.db.base.Database.objects`


        Returns
        -------

            probes : list
                List of samples conforming the pipeline API for the creation of
                biometric probes.  See, e.g., :py:func:`.pipelines.first`.

        """

        probes = dict()

        for m in self.database.model_ids(protocol=self.protocol, groups=group):

            # Getting all the probe objects from a particular biometric
            # reference
            objects = self.database.objects(
                protocol=self.protocol, groups=group, model_ids=(m,), purposes="probe"
            )

            # Creating probe samples
            for o in objects:
                if o.id not in probes:
                    probes[o.id] = SampleSet(
                        [
                            DelayedSample(
                                load=functools.partial(
                                    o.load,
                                    self.database.original_directory,
                                    self.database.original_extension,
                                ),
                                id=o.id,
                                path=o.path,
                            )
                        ],
                        id=o.id,
                        path=o.path,
                        subject=o.client_id,
                        references=[m],
                    )
                else:
                    probes[o.id].references.append(m)

        return list(probes.values())


class AlgorithmAdaptor:
    """Describes a biometric model based on :py:class:`bob.bio.base.algorithm.Algorithm`'s

    The model can be fitted (optionally).  Otherwise, it can only execute
    biometric model enrollement, via ``enroll()`` and scoring, with
    ``score()``.

    Parameters
    ----------

        algorithm : object
            An object that can be initialized by default and posseses the
            following attributes and methods:

            * attribute ``requires_projector_training``: indicating if the
              model is fittable or not
            * method ``train_projector(samples, path)``: receives a list of
              objects produced by the equivalent ``Sample.data`` object, fed
              **after** sample loading by the equivalent pipeline, and records
              the model to an on-disk file
            * method ``load_projector(path)``: loads the model state from a file
            * method ``project(sample)``: projects the data to an embedding
              from a single sample
            * method ``enroll(samples)``: creates a scorable biometric
              reference from a set of input samples
            * method ``score(model, probe)``: scores a single probe, given the
              input model, which can be obtained by a simple
              ``project(sample)``

            If the algorithm cannot be initialized by default, pass the result
            of :py:func:`functools.partial` instead.

        path : string
            A path leading to a place where to save the fitted model or, in
            case this model is not fittable (``not is_fitable == False``), then
            name of the model to load for running prediction and scoring.

    """

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.extension = ".hdf5"

    def fit(self, samplesets, checkpoint):
        """Fits this model, if it is fittable

        Parameters
        ----------

            samplesets : list
                A list of :py:class:`SampleSet`s to be used for fitting this
                model

            checkpoint : str
                If provided, must the path leading to a location where this
                model should be saved at (complete path without extension) -
                currently, it needs to be provided because of existing
                serialization requirements (see bob/bob.io.base#106), but
                checkpointing will still work as expected.


        Returns
        -------

            model : str
                A path leading to the fitted model

        """

        self.path = checkpoint + self.extension
        if not os.path.exists(self.path):  # needs training
            model = self.algorithm()
            bob.io.base.create_directories_safe(os.path.dirname(self.path))
            if model.requires_projector_training:
                alldata = [
                    sample.data
                    for sampleset in samplesets
                    for sample in sampleset.samples
                ]
                model.train_projector(alldata, self.path)

        return self.path

    def enroll(self, references, path, checkpoint, *args, **kwargs):
        """Runs prediction on multiple input samples

        This method is optimized to deal with multiple reference biometric
        samples at once, organized in partitions


        Parameters
        ----------

            references : list
                A list of :py:class:`SampleSet` objects to be used for
                creating biometric references.  The sets must be identified
                with a unique id and a path, for eventual checkpointing.

            path : str
                Path pointing to stored model on disk

            checkpoint : str, None
                If passed and not ``None``, then it is considered to be the
                path of a directory containing possible cached values for each
                of the references in this experiment.  If that is the case, the
                values are loaded from there and not recomputed.

            *args, **kwargs :
                Extra parameters that can be used to hook-up processing graph
                dependencies, but are currently ignored

        Returns
        -------

            references : list
                A list of :py:class:`.samples.Reference` objects that can be
                used in scoring

        """

        class _CachedModel:
            def __init__(self, algorithm, path):
                self.model = algorithm()
                self.loaded = False
                self.path = path

            def load(self):
                if not self.loaded:
                    self.model.load_projector(self.path)
                    self.loaded = True

            def enroll(self, k):
                self.load()
                if self.model.requires_projector_training:
                    return self.model.enroll(
                        numpy.array([self.model.project(s.data) for s in k.samples])
                    )
                else:
                    return self.model.enroll(numpy.array([s.data for s in k.samples]))

            def write_enrolled(self, k, path):
                self.model.write_model(k, path)

        model = _CachedModel(self.algorithm, path)

        retval = []        
        for k in references:
            if checkpoint is not None:
                candidate = os.path.join(os.path.join(checkpoint, k.path + ".hdf5"))
                if not os.path.exists(candidate):
                    # create new checkpoint
                    bob.io.base.create_directories_safe(os.path.dirname(candidate))
                    enrolled = model.enroll(k)
                    model.model.write_model(enrolled, candidate)                
                retval.append(
                    DelayedSample(
                        functools.partial(model.model.read_model, candidate), parent=k
                    )
                )
            else:
                # compute on-the-fly
                retval.append(Sample(model.enroll(k), parent=k))
        return retval

    def score(self, probes, references, path, *args, **kwargs):
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

            path : str
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

        model = self.algorithm()
        model.load_projector(path)

        score_sample_sets = []
        n_probes = len(probes)

        for i,p in enumerate(probes):
            if model.requires_projector_training:
                data = [model.project(s.data) for s in p.samples]
            else:
                data = [s.data for s in p.samples]

            for subprobe_id, (s, parent) in enumerate(zip(data, p.samples)):

                # each sub-probe in the probe needs to be checked
                subprobe_scores = []
                def _compute_score(ref, probe_sample):
                    return Sample(model.score(ref.data, probe_sample), parent=ref)

                # Parellelizing the scoring
                subprobe_scores_delayed = []
                for ref in [r for r in references if r.id in p.references]:
                    # Delaying the computation of a single score.
                    subprobe_scores_delayed.append(dask.delayed(_compute_score)(ref, s))
                    #subprobe_scores.append(Sample(model.score(ref.data, s), parent=ref))
                #subprobe_scores = [ssd.compute() for ssd in subprobe_scores_delayed]
                
                # Delaying the computation of a single score.
                subprobe_scores = dask.delayed(list)(subprobe_scores_delayed)
                subprobe = SampleSet(subprobe_scores, parent=parent)
                #subprobe = SampleSet(subprobe_scores, parent=p)
                #subprobe = SampleSet(subprobe_scores, parent=None)

                subprobe.subprobe_id = subprobe_id
                score_sample_sets.append(subprobe)


        return score_sample_sets
