#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Re-usable blocks for legacy bob.bio.base algorithms"""

import functools
import logging
import os

import joblib
from bob.bio.base.algorithm import Algorithm
from bob.pipelines import DelayedSample
from bob.pipelines import DelayedSampleSet
from bob.pipelines import SampleSet
from bob.db.base.utils import (
    check_parameters_for_validity,
    convert_names_to_highlevel,
    convert_names_to_lowlevel,
)
import pickle
from .abstract_classes import BioAlgorithm
from .abstract_classes import Database
import tempfile

logger = logging.getLogger("bob.bio.base")


def get_temp_directory(sub_dir):
    """
    Get Temporary directory for legacy algorithms.
    Most of the legacy algorithms are not pickled serializable, therefore,
    we can't wrap their transformations as :any:`bob.pipelines.Sample`.
    Hence, we need to make them automatically checkpointable.
    This function returns possible temporary directories to store such checkpoints.
    The possible temporary directorys are (in order of priority):
       1. `/.../temp/[user]/bob.bio.base.legacy_cache` if you are at Idiap and has acess to temp.
       2. `~/temp/` in case the algorithm runs in the CI
       3. `/tmp/bob.bio.base.legacy_cache` in case your are not at Idiap

    Parameters
    ----------

    sub_dir: str
        Sub-directory to checkpoint your data

    """

    default_temp = (
        os.path.join("/idiap", "temp", os.environ["USER"])
        if "USER" in os.environ
        else "~/temp"
    )
    if os.path.exists(default_temp):
        return os.path.join(default_temp, "bob.bio.base.legacy_cache", sub_dir)
    else:
        # if /idiap/temp/<USER> does not exist, use /tmp/tmpxxxxxxxx
        return os.path.join(tempfile.TemporaryDirectory().name, sub_dir)


def _biofile_to_delayed_sample(biofile, database):
    return DelayedSample(
        load=functools.partial(
            biofile.load, database.original_directory, database.original_extension,
        ),
        reference_id=str(biofile.client_id),
        key=biofile.path,
        path=biofile.path,
        delayed_attributes=dict(
            annotations=functools.partial(database.annotations, biofile)
        ),
    )


class DatabaseConnector(Database):
    """Wraps a legacy bob.bio.base database and generates conforming samples

    This connector allows wrapping generic bob.bio.base datasets and generate
    samples that conform to the specifications of biometric pipelines defined
    in this package.


    Parameters
    ----------

    database : object
        An instantiated version of a bob.bio.base.Database object

    protocol : str
        The name of the protocol to generate samples from.
        To be plugged at `bob.db.base.Database.objects`.

    allow_scoring_with_all_biometric_references: bool
        If True will allow the scoring function to be performed in one shot with multiple probes.
        This optimization is useful when all probes needs to be compared with all biometric references AND
        your scoring function allows this broadcast computation.

    annotation_type: str
        Type of the annotations that the database provide.
        Allowed types are: `eyes-center` and `bounding-box`

    fixed_positions: dict
        In case database contains one single annotation for all samples.
        This is useful for registered databases.

    memory_demanding: bool
        Sinalizes that a database has some memory demanding components.
        It might be useful for future processing
    """

    def __init__(
        self,
        database,
        allow_scoring_with_all_biometric_references=True,
        annotation_type="eyes-center",
        fixed_positions=None,
        memory_demanding=False,
        **kwargs,
    ):
        self.database = database
        self.allow_scoring_with_all_biometric_references = (
            allow_scoring_with_all_biometric_references
        )
        self.annotation_type = annotation_type
        self.fixed_positions = fixed_positions
        self.memory_demanding = memory_demanding

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
                    reference_id=(str(m)),
                    subject_id=str(self.database.client_id_from_model_id(m)),
                )
            )

        return retval

    def probes(self, group="dev"):
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
                        key=str(o.path),
                        path=o.path,
                        reference_id=str(m),
                        references=[str(m)],
                        subject_id=o.client_id,
                    )
                else:
                    probes[o.id].references.append(str(str(m)))
        return list(probes.values())

    def all_samples(self, groups=None):
        """Returns all the legacy database files in Sample format

        Parameters
        ----------
        groups: list or `None`
            List of groups to consider ('train', 'dev', and/or 'eval').
            If `None` is given, returns samples from all the groups.

        Returns
        -------
        samples: list
            List of all the samples of a database in :class:`bob.pipelines.Sample`
            objects.
        """
        valid_groups = convert_names_to_highlevel(
            self.database.groups(),
            low_level_names=["world", "dev", "eval"],
            high_level_names=["train", "dev", "eval"],
        )
        groups = check_parameters_for_validity(
            parameters=groups,
            parameter_description="groups",
            valid_parameters=valid_groups,
            default_parameters=valid_groups,
        )
        logger.debug(f"Fetching all samples of groups '{groups}'.")
        low_level_groups = convert_names_to_lowlevel(
            names=groups,
            low_level_names=["world", "dev", "eval"],
            high_level_names=["train", "dev", "eval"],
        )
        objects = self.database.all_files(groups=low_level_groups)
        return [_biofile_to_delayed_sample(k, self.database) for k in objects]


class BioAlgorithmLegacy(BioAlgorithm):
    """Biometric Algorithm that handles :py:class:`bob.bio.base.algorithm.Algorithm`

    In this design, :any:`BioAlgorithm.enroll` maps to :py:meth:`bob.bio.base.algorithm.Algorithm.enroll`
    and :any:`BioAlgorithm.score` maps to :py:meth:`bob.bio.base.algorithm.Algorithm.score`

    .. note::
        Legacy algorithms are always checkpointable


    Parameters
    ----------
      instance: object
         An instance of :py:class:`bob.bio.base.algorithm.Algorithm`


    Example
    -------
        >>> from bob.bio.base.pipelines.vanilla_biometrics import BioAlgorithmLegacy
        >>> from bob.bio.base.algorithm import PCA
        >>> biometric_algorithm = BioAlgorithmLegacy(PCA(subspace_dimension=0.99), base_dir="./", projector_file="Projector.hdf5")

    """

    def __init__(
        self, instance, base_dir, force=False, projector_file=None, **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(instance, Algorithm):
            raise ValueError(
                f"Only `bob.bio.base.Algorithm` supported, not `{instance}`"
            )
        logger.info(f"Using `bob.bio.base` legacy algorithm {instance}")

        if instance.requires_projector_training and projector_file is None:
            raise ValueError(f"{instance} requires a `projector_file` to be set")

        self.instance = instance
        self.is_background_model_loaded = False

        self.projector_file = projector_file
        self.biometric_reference_dir = os.path.join(base_dir, "biometric_references")
        self._biometric_reference_extension = ".hdf5"
        self.score_dir = os.path.join(base_dir, "scores")
        self.force = force
        self._base_dir = base_dir
        self._score_extension = ".joblib"

    @property
    def base_dir(self):
        return self._base_dir

    @base_dir.setter
    def base_dir(self, v):
        self._base_dir = v
        self.biometric_reference_dir = os.path.join(
            self._base_dir, "biometric_references"
        )
        self.score_dir = os.path.join(self._base_dir, "scores")
        if self.projector_file is not None:
            self.projector_file = os.path.join(
                self._base_dir, os.path.basename(self.projector_file)
            )

    def load_legacy_background_model(self):
        # Loading background model
        if not self.is_background_model_loaded:
            self.instance.load_projector(self.projector_file)
            self.is_background_model_loaded = True

    def enroll(self, enroll_features, **kwargs):
        self.load_legacy_background_model()
        return self.instance.enroll(enroll_features)

    def score(self, biometric_reference, data, **kwargs):
        self.load_legacy_background_model()
        scores = self.instance.score(biometric_reference, data)
        if isinstance(scores, list):
            scores = self.instance.probe_fusion_function(scores)
        return scores

    def score_multiple_biometric_references(self, biometric_references, data, **kwargs):
        scores = self.instance.score_for_multiple_models(biometric_references, data)
        return scores

    def write_biometric_reference(self, sample, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.instance.write_model(sample.data, path)

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
            functools.partial(self.instance.read_model, path), parent=sampleset
        )

        return delayed_enrolled_sample

    def write_scores(self, samples, path):
        try:
            final_path = os.path.dirname(path)
            os.makedirs(final_path, exist_ok=True)
        except NotADirectoryError:
            # If it cannot make the directory, saves in /tmp/
            # This is useful and necessary for some
            # very specific types of baselines (ROC)
            final_path = "/tmp/" + os.path.dirname(path)
            os.makedirs(final_path, exist_ok=True)
            path = "/tmp/" + path
        open(path, "wb").write(pickle.dumps(samples))
        joblib.dump(samples, path, compress=4)

    def _score_sample_set(
        self,
        sampleset,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):
        def _load(path):
            # return pickle.loads(open(path, "rb").read())
            if os.path.exists(path):
                return joblib.load(path)
            else:
                # If doesn't exists. the cached sample might be at /tmp
                return joblib.load("/tmp/" + path)

        def _make_name(sampleset, biometric_references):
            # The score file name is composed by sampleset key and the
            # first 3 biometric_references
            name = str(sampleset.key)
            suffix = "_".join([str(s.key) for s in biometric_references[0:3]])
            return name + suffix

        path = os.path.join(
            self.score_dir,
            _make_name(sampleset, biometric_references) + self._score_extension,
        )

        if self.force or not os.path.exists(path):

            # Computing score
            scored_sample_set = super()._score_sample_set(
                sampleset,
                biometric_references,
                allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
            )

            self.write_scores(scored_sample_set.samples, path)

            scored_sample_set = DelayedSampleSet(
                functools.partial(_load, path), parent=scored_sample_set
            )

        else:
            scored_sample_set = SampleSet(_load(path), parent=sampleset)

        return scored_sample_set
