#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Re-usable blocks for legacy bob.bio.base algorithms"""

import os
import functools
from collections import defaultdict

from bob.bio.base import utils
from .abstract_classes import (
    BioAlgorithm,
    Database,
    create_score_delayed_sample,
    make_four_colums_score,
)
from bob.io.base import HDF5File
from bob.pipelines.mixins import SampleMixin, CheckpointMixin
from bob.pipelines.sample import DelayedSample, SampleSet, Sample
from sklearn.base import TransformerMixin, BaseEstimator
import logging
import copy


logger = logging.getLogger("bob.bio.base")


def _biofile_to_delayed_sample(biofile, database):
    return DelayedSample(
        load=functools.partial(
            biofile.load, database.original_directory, database.original_extension,
        ),
        subject=str(biofile.client_id),
        key=biofile.path,
        path=biofile.path,
        annotations=database.annotations(biofile),
    )


class DatabaseConnector(Database):
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

    def __init__(self, database, **kwargs):
        self.database = database

    def background_model_samples(self):
        """Returns :py:class:`Sample`'s to train a background model (group
        ``world``).


        Returns
        -------

            samples : list
                List of samples conforming the pipeline API for background
                model training.  See, e.g., :py:func:`.pipelines.first`.

        """
        objects = self.database.training_files()
        return [_biofile_to_delayed_sample(k, self.database) for k in objects]

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
        for m in self.database.model_ids(groups=group):

            objects = self.database.enroll_files(group=group, model_id=m)

            retval.append(
                SampleSet(
                    [_biofile_to_delayed_sample(k, self.database) for k in objects],
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

        for m in self.database.model_ids(groups=group):

            # Getting all the probe objects from a particular biometric
            # reference
            objects = self.database.probe_files(group=group, model_id=m)
            # Creating probe samples
            for o in objects:
                if o.id not in probes:
                    probes[o.id] = SampleSet(
                        [_biofile_to_delayed_sample(o, self.database)],
                        key=str(o.client_id),
                        path=o.path,
                        subject=str(o.client_id),
                        references=[str(m)],
                    )
                else:
                    probes[o.id].references.append(str(m))

        return list(probes.values())


class AlgorithmAsBioAlg(_NonPickableWrapper, BioAlgorithm):
    """Biometric Algorithm that handles legacy :py:class:`bob.bio.base.algorithm.Algorithm`


    :py:method:`BioAlgorithm.enroll` maps to :py:method:`bob.bio.base.algorithm.Algoritm.enroll`

    :py:method:`BioAlgorithm.score` maps :py:method:`bob.bio.base.algorithm.Algoritm.score`


    Example
    -------


    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the bob.bio.base.algorithm.Algorithm

    """

    def __init__(
        self, callable, features_dir, extension=".hdf5", model_path=None, **kwargs
    ):
        super().__init__(callable, **kwargs)
        self.features_dir = features_dir
        self.biometric_reference_dir = os.path.join(
            self.features_dir, "biometric_references"
        )
        self.score_dir = os.path.join(self.features_dir, "scores")
        self.extension = extension
        self.model_path = model_path
        self.is_projector_loaded = False

    def _enroll_sample_set(self, sampleset):
        # Enroll
        return self.enroll(sampleset)

    def _load_projector(self):
        """
        Run :py:meth:`bob.bio.base.algorithm.Algorithm.load_projector` if necessary by
        :py:class:`bob.bio.base.algorithm.Algorithm`
        """
        if self.instance.performs_projection and not self.is_projector_loaded:
            if self.model_path is None:
                raise ValueError(
                    "Algorithm " + f"{self. instance} performs_projection. Hence, "
                    "`model_path` needs to passed in `AlgorithmAsBioAlg.__init__`"
                )
            else:
                # Loading model
                self.instance.load_projector(self.model_path)
                self.is_projector_loaded = True

    def _restore_state_of_ref(self, ref):
        """
        There are some algorithms that :py:meth:`bob.bio.base.algorithm.Algorithm.read_model` or 
        :py:meth:`bob.bio.base.algorithm.Algorithm.read_feature` depends
        on the state of `self` to be properly loaded.
        In these cases, it's not possible to rely only in the unbounded method extracted by
        :py:func:`_get_pickable_method`.

        This function replaces the current state of these objects (that are not)
        by bounding them with `self.instance`
        """
        
        if isinstance(ref, DelayedSample):
            new_ref = copy.copy(ref)            
            new_ref.load = functools.partial(ref.load.func, self.instance, ref.load.args[1])
            #new_ref.load = functools.partial(ref.load.func, self.instance, ref.load.args[1])
            return new_ref
        else:
            return ref

    def _score_sample_set(
        self,
        sampleset,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):
        """Given a sampleset for probing, compute the scores and retures a sample set with the scores
        """

        # Compute scores for each sample inside of the sample set
        # TODO: In some cases we want to compute 1 score per sampleset (IJB-C)
        # We should add an agregator function here so we can properlly agregate samples from
        # a sampleset either after or before scoring.
        # To be honest, this should be the default behaviour

        def _write_sample(ref, probe, score):
            data = make_four_colums_score(ref.subject, probe.subject, probe.path, score)
            return Sample(data, parent=ref)

        self._load_projector()
        retval = []

        for subprobe_id, s in enumerate(sampleset.samples):
            # Creating one sample per comparison
            subprobe_scores = []

            if allow_scoring_with_all_biometric_references:
                if self.stacked_biometric_references is None:
                    self.stacked_biometric_references = [
                        ref.data for ref in biometric_references
                    ]

                #s = self._restore_state_of_ref(s)
                scores = self.score_multiple_biometric_references(
                    self.stacked_biometric_references, s.data
                )
                # Wrapping the scores in samples
                for ref, score in zip(biometric_references, scores):
                    subprobe_scores.append(_write_sample(ref, sampleset, score))

            else:
                for ref in [
                    r for r in biometric_references if r.key in sampleset.references
                ]:
                    
                    score = self.score(ref.data, s.data)
                    subprobe_scores.append(_write_sample(ref, sampleset, score))

            # Creating one sampleset per probe
            subprobe = SampleSet(subprobe_scores, parent=sampleset)
            subprobe.subprobe_id = subprobe_id

            # Checkpointing score MANDATORY FOR LEGACY
            path = os.path.join(self.score_dir, str(subprobe.path) + ".txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            delayed_scored_sample = create_score_delayed_sample(path, subprobe)
            subprobe.samples = [delayed_scored_sample]

            retval.append(subprobe)

        return retval

    def enroll(self, enroll_features, **kwargs):

        if not isinstance(enroll_features, SampleSet):
            raise ValueError(
                f"`enroll_features` should be the type SampleSet, not {enroll_features}"
            )

        path = os.path.join(
            self.biometric_reference_dir, str(enroll_features.key) + self.extension
        )
        self._load_projector()
        if path is None or not os.path.isfile(path):
            # Enrolling
            data = [s.data for s in enroll_features.samples]
            model = self.instance.enroll(data)

            # Checkpointing
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.instance.write_model(model, path)
        
        reader = self.instance.read_model
        return  DelayedSample(functools.partial(reader, path), parent=enroll_features)

    def score(self, biometric_reference, data, **kwargs):
        return self.instance.score(biometric_reference, data)

    def score_multiple_biometric_references(self, biometric_references, data, **kwargs):
        """
        It handles the score computation of one probe against multiple biometric references using legacy
        `bob.bio.base`

        Basically it wraps :py:meth:`bob.bio.base.algorithm.Algorithm.score_for_multiple_models`.

        """        
        scores = self.instance.score_for_multiple_models(biometric_references, data)
        return scores
