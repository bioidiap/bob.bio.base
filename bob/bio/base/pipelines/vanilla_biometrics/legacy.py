#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Re-usable blocks for legacy bob.bio.base algorithms"""

import os
import copy
import functools

import bob.io.base
from bob.pipelines.sample import DelayedSample, SampleSet, Sample
import numpy
import logging
import dask
import sys
import pickle
from bob.bio.base.mixins.legacy import get_reader
from .biometric_algorithm import save_scores_four_columns

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
                        key=k.path,
                        path=k.path,
                    )
                ],
                key=str(k.client_id),
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
        for m in self.database.model_ids_with_protocol(protocol=self.protocol, groups=group):

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
                            key=k.path,
                            path=k.path,
                        )
                        for k in objects
                    ],
                    key=str(m),
                    path=str(m),
                    subject=str(objects[0].client_id),
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

        for m in self.database.model_ids_with_protocol(protocol=self.protocol, groups=group):

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
                                key=o.path,
                                path=o.path,
                            )
                        ],
                        key=str(o.client_id),
                        path=o.path,
                        subject=str(o.client_id),
                        references=[str(m)],
                    )
                else:
                    probes[o.id].references.append(m)

        return list(probes.values())



def _load_data_and_annotations(bio_file, annotations, original_directory, original_extension):
    """
    Return a tuple (data, annotations) given a :py:class:`bob.bio.base.database.BioFile` as input

    Parameters
    ----------

     bio_file: :py:class:`bob.bio.base.database.BioFile`
        Input bio file

    Returns
    -------
        (data, annotations): A dictionary containing the raw data + annotations

    """

    data = bio_file.load(original_directory, original_extension)

    # I know it sounds stupid to return the the annotations here without any transformation
    # but I can't do `database.annotations(bio_file)`, SQLAlcheamy session is not picklable
    return {"data": data, "annotations": annotations}


class DatabaseConnectorAnnotated(DatabaseConnector):
    """Wraps a bob.bio.base database and generates conforming samples for datasets
    that has annotations

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
        super(DatabaseConnectorAnnotated, self).__init__(database, protocol)


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
                            _load_data_and_annotations, k, self.database.annotations(k), self.database.original_directory, self.database.original_extension
                        ),
                        key=k.path,
                        path=k.path,
                        annotations=self.database.annotations(k),
                    )
                ],
                key=str(k.client_id),
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

        for m in self.database.model_ids_with_protocol(
            protocol=self.protocol, groups=group
        ):

            objects = self.database.objects(
                protocol=self.protocol, groups=group, model_ids=(m,), purposes="enroll"
            )

            retval.append(
                SampleSet(
                    [
                        DelayedSample(
                            load=functools.partial(
                                _load_data_and_annotations, k, self.database.annotations(k), self.database.original_directory, self.database.original_extension
                            ),
                            key=k.path,
                            path=k.path,
                            subject=str(objects[0].client_id),
                            annotations=self.database.annotations(k),
                        )
                        for k in objects
                    ],
                    key=str(m),
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

        for m in self.database.model_ids_with_protocol(
            protocol=self.protocol, groups=group
        ):

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
                                    _load_data_and_annotations, o, self.database.annotations(o), self.database.original_directory, self.database.original_extension
                                ),
                                key=o.path,
                                path=o.path,
                                annotations=self.database.annotations(o),
                            )
                        ],
                        key=str(o.client_id),
                        path=o.path,
                        subject=o.client_id,
                        references=[str(m)],
                    )
                else:
                    probes[o.id].references.append(str(m))

        return list(probes.values())


from .biometric_algorithm import BiometricAlgorithm


class LegacyBiometricAlgorithm(BiometricAlgorithm):
    """Biometric Algorithm that handles legacy :py:class:`bob.bio.base.algorithm.Algorithm`


    :py:method:`BiometricAlgorithm.enroll` maps to :py:method:`bob.bio.base.algorithm.Algoritm.enroll`

    :py:method:`BiometricAlgorithm.score` maps :py:method:`bob.bio.base.algorithm.Algoritm.score`


    THIS CODE HAS TO BE CHECKPOINTABLE IN A SPECIAL WAY

    Example
    -------


    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the scikit estimator

    """

    def __init__(self, callable=None, features_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.callable = callable
        self.instance = None
        self.projector_file = None
        self.features_dir = features_dir
        self.biometric_reference_dir = os.path.join(
            self.features_dir, "biometric_references"
        )
        self.score_dir = os.path.join(self.features_dir, "scores")
        self.extension = ".hdf5"

    def _enroll_sample_set(self, sampleset):
        # Enroll
        return self.enroll(sampleset)

    def _score_sample_set(self, sampleset, biometric_references, extractor):
        """Given a sampleset for probing, compute the scores and retures a sample set with the scores
        """

        # Stacking the samples from a sampleset
        data = [s for s in sampleset.samples]

        # Compute scores for each sample inside of the sample set
        # TODO: In some cases we want to compute 1 score per sampleset (IJB-C)
        # We should add an agregator function here so we can properlly agregate samples from
        # a sampleset either after or before scoring.
        # To be honest, this should be the default behaviour
        retval = []
        for subprobe_id, s in enumerate(sampleset.samples):
            # Creating one sample per comparison
            subprobe_scores = []

            for ref in [
                r for r in biometric_references if r.key in sampleset.references
            ]:
                # subprobe_scores.append(self.score(ref.data, s, extractor))
                subprobe_scores.append(
                    Sample(self.score(ref.data, s.data, extractor), parent=ref)
                )

            # Creating one sampleset per probe
            subprobe = SampleSet(subprobe_scores, parent=sampleset)
            subprobe.subprobe_id = subprobe_id

            # Checkpointing score MANDATORY FOR LEGACY
            path = os.path.join(self.score_dir, str(subprobe.path) + ".txt")
            bob.io.base.create_directories_safe(os.path.dirname(path))

            delayed_scored_sample = save_scores_four_columns(path, subprobe)
            subprobe.samples = [delayed_scored_sample]

            retval.append(subprobe)

        return retval

    def enroll(self, enroll_features, **kwargs):

        if not isinstance(enroll_features, SampleSet):
            raise ValueError(
                f"`enroll_features` should be the type SampleSet, not {enroll_features}"
            )

        # Instantiates and do the "real" fit
        if self.instance is None:
            self.instance = self.callable()

        path = os.path.join(
            self.biometric_reference_dir, str(enroll_features.key) + self.extension
        )
        if path is None or not os.path.isfile(path):

            # Enrolling
            data = [s.data for s in enroll_features.samples]
            model = self.instance.enroll(data)

            # Checkpointing
            bob.io.base.create_directories_safe(os.path.dirname(path))
            hdf5 = bob.io.base.HDF5File(path, "w")
            self.instance.write_model(model, hdf5)

        reader = get_reader(self.instance.read_model, path)
        return DelayedSample(reader, parent=enroll_features)

    def score(self, model, probe, extractor=None, **kwargs):

        # Instantiates and do the "real" fit
        if self.instance is None:
            self.instance = self.callable()

        return self.instance.score(model, probe)
