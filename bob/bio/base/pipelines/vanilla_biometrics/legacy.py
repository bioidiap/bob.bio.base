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
)
from bob.io.base import HDF5File
from bob.pipelines import DelayedSample, SampleSet, Sample
import logging
import copy

from .score_writers import FourColumnsScoreWriter

from bob.bio.base.algorithm import Algorithm

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


class BioAlgorithmLegacy(BioAlgorithm):
    """Biometric Algorithm that handlesy :any:`bob.bio.base.algorithm.Algorithm`

    In this design, :any:`BioAlgorithm.enroll` maps to :any:`bob.bio.base.algorithm.Algorithm.enroll` and 
    :any:`BioAlgorithm.score` maps :any:`bob.bio.base.algorithm.Algorithm.score`
    
    .. note::
        Legacy algorithms are always checkpointable     


    Parameters
    ----------
      callable: ``collection.callable``
         Callable function that instantiates the :any:`bob.bio.base.algorithm.Algorithm`


    Example
    -------
        >>> from bob.bio.base.pipelines.vanilla_biometrics import BioAlgorithmLegacy
        >>> from bob.bio.base.algorithm import PCA
        >>> biometric_algorithm = BioAlgorithmLegacy(PCA())

    """

    def __init__(
        self,
        callable,
        base_dir,
        force=False,
        projector_file=None,
        score_writer=FourColumnsScoreWriter(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(callable, Algorithm):
            raise ValueError(
                f"Only `bob.bio.base.Algorithm` supported, not `{callable}`"
            )
        logger.info(f"Using `bob.bio.base` legacy algorithm {callable}")

        if callable.requires_projector_training and projector_file is None:
            raise ValueError(f"{callable} requires a `projector_file` to be set")

        self.callable = callable
        self.is_background_model_loaded = False

        self.projector_file = projector_file
        self.biometric_reference_dir = os.path.join(base_dir, "biometric_references")
        self._biometric_reference_extension = ".hdf5"
        self.score_dir = os.path.join(base_dir, "scores")
        self.score_writer = score_writer
        self.force = force

    def load_legacy_background_model(self):
        # Loading background model
        if not self.is_background_model_loaded:
            self.callable.load_projector(self.projector_file)
            self.is_background_model_loaded = True

    def enroll(self, enroll_features, **kwargs):
        self.load_legacy_background_model()
        return self.callable.enroll(enroll_features)

    def score(self, biometric_reference, data, **kwargs):
        self.load_legacy_background_model()
        scores = self.callable.score(biometric_reference, data)
        if isinstance(scores, list):
            scores = self.callable.probe_fusion_function(scores)
        return scores

    def score_multiple_biometric_references(self, biometric_references, data, **kwargs):
        scores = self.callable.score_for_multiple_models(biometric_references, data)
        return scores

    def write_biometric_reference(self, sample, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.callable.write_model(sample.data, path)

    def _enroll_sample_set(self, sampleset):
        """
        Enroll a sample set with checkpointing
        """
        # Amending `models` directory
        path = os.path.join(
            self.biometric_reference_dir,
            str(sampleset.key) + self._biometric_reference_extension,
        )

        if self.force or not os.path.exists(path):
            enrolled_sample = super()._enroll_sample_set(sampleset)

            # saving the new sample
            self.write_biometric_reference(enrolled_sample, path)

        delayed_enrolled_sample = DelayedSample(
            functools.partial(self.callable.read_model, path), parent=sampleset
        )

        return delayed_enrolled_sample

    def _score_sample_set(
        self,
        sampleset,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):
        path = os.path.join(self.score_dir, str(sampleset.key))
        # Computing score
        scored_sample_set = super()._score_sample_set(
            sampleset,
            biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        scored_sample_set = self.score_writer.write(scored_sample_set, path)

        return scored_sample_set
