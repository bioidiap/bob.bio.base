#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Re-usable blocks for legacy bob.bio.base algorithms"""

import os
import functools
from collections import defaultdict

from bob.bio.base import utils
from .abstract_classes import BioAlgorithm, Database, save_scores_four_columns
from bob.io.base import HDF5File
from bob.pipelines.mixins import SampleMixin, CheckpointMixin
from bob.pipelines.sample import DelayedSample, SampleSet, Sample
from bob.pipelines.utils import is_picklable
from sklearn.base import TransformerMixin
import logging

logger = logging.getLogger("bob.bio.base")


def _biofile_to_delayed_sample(biofile, database):
    return DelayedSample(
        load=functools.partial(
            biofile.load, database.original_directory, database.original_extension,
        ),
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


class _NonPickableWrapper:
    def __init__(self, callable, **kwargs):
        super().__init__(**kwargs)
        self.callable = callable
        self._instance = None

    @property
    def instance(self):
        if self._instance is None:
            self._instance = self.callable()
        return self._instance

    def __setstate__(self, d):
        # Handling unpicklable objects
        self._instance = None
        return super().__setstate__(d)

    def __getstate__(self):
        # Handling unpicklable objects
        self._instance = None
        return super().__getstate__()


class _Preprocessor(_NonPickableWrapper, TransformerMixin):
    def transform(self, X, annotations):
        return [self.instance(data, annot) for data, annot in zip(X, annotations)]

    def _more_tags(self):
        return {"stateless": True}


def _get_pickable_method(method):
    if not is_picklable(method):
        logger.warning(
            f"The method {method} is not picklable. Returning its unbounded method"
        )
        method = functools.partial(method.__func__, None)
    return method


class Preprocessor(CheckpointMixin, SampleMixin, _Preprocessor):
    def __init__(self, callable, **kwargs):
        instance = callable()
        super().__init__(
            callable=callable,
            transform_extra_arguments=(("annotations", "annotations"),),
            load_func=_get_pickable_method(instance.read_data),
            save_func=_get_pickable_method(instance.write_data),
            **kwargs,
        )


def _split_X_by_y(X, y):
    training_data = defaultdict(list)
    for x1, y1 in zip(X, y):
        training_data[y1].append(x1)
    training_data = training_data.values()
    return training_data


class _Extractor(_NonPickableWrapper, TransformerMixin):
    def transform(self, X, metadata=None):
        if self.requires_metadata:
            return [self.instance(data, metadata=m) for data, m in zip(X, metadata)]
        else:
            return [self.instance(data) for data in X]

    def fit(self, X, y=None):
        if not self.instance.requires_training:
            return self

        training_data = X
        if self.instance.split_training_data_by_client:
            training_data = _split_X_by_y(X, y)

        self.instance.train(self, training_data, self.model_path)
        return self

    def _more_tags(self):
        return {"requires_fit": self.instance.requires_training}


class Extractor(CheckpointMixin, SampleMixin, _Extractor):
    def __init__(self, callable, model_path, **kwargs):
        instance = callable()

        transform_extra_arguments = None
        self.requires_metadata = False
        if utils.is_argument_available("metadata", instance.__call__):
            transform_extra_arguments = (("metadata", "metadata"),)
            self.requires_metadata = True

        fit_extra_arguments = None
        if instance.requires_training and instance.split_training_data_by_client:
            fit_extra_arguments = (("y", "subject"),)

        super().__init__(
            callable=callable,
            transform_extra_arguments=transform_extra_arguments,
            fit_extra_arguments=fit_extra_arguments,
            load_func=_get_pickable_method(instance.read_feature),
            save_func=_get_pickable_method(instance.write_feature),
            model_path=model_path,
            **kwargs,
        )

    def load_model(self):
        self.instance.load(self.model_path)
        return self

    def save_model(self):
        # we have already saved the model in .fit()
        return self


class _AlgorithmTransformer(_NonPickableWrapper, TransformerMixin):
    def transform(self, X):
        return [self.instance.project(feature) for feature in X]

    def fit(self, X, y=None):
        if not self.instance.requires_projector_training:
            return self

        training_data = X
        if self.instance.split_training_features_by_client:
            training_data = _split_X_by_y(X, y)

        self.instance.train_projector(self, training_data, self.model_path)
        return self

    def _more_tags(self):
        return {"requires_fit": self.instance.requires_projector_training}


class AlgorithmAsTransformer(CheckpointMixin, SampleMixin, _AlgorithmTransformer):
    """Class that wraps :py:class:`bob.bio.base.algorithm.Algoritm`

    :py:method:`LegacyAlgorithmrMixin.fit` maps to :py:method:`bob.bio.base.algorithm.Algoritm.train_projector`

    :py:method:`LegacyAlgorithmrMixin.transform` maps :py:method:`bob.bio.base.algorithm.Algoritm.project`

    Example
    -------

        Wrapping LDA algorithm with functtools
        >>> from bob.bio.base.pipelines.vanilla_biometrics.legacy import LegacyAlgorithmAsTransformer
        >>> from bob.bio.base.algorithm import LDA
        >>> import functools
        >>> transformer = LegacyAlgorithmAsTransformer(functools.partial(LDA, use_pinv=True, pca_subspace_dimension=0.90))



    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the bob.bio.base.algorithm.Algorithm

    """

    def __init__(self, callable, model_path, **kwargs):
        instance = callable()

        fit_extra_arguments = None
        if (
            instance.requires_projector_training
            and instance.split_training_features_by_client
        ):
            fit_extra_arguments = (("y", "subject"),)

        super().__init__(
            callable=callable,
            fit_extra_arguments=fit_extra_arguments,
            load_func=_get_pickable_method(instance.read_feature),
            save_func=_get_pickable_method(instance.write_feature),
            model_path=model_path,
            **kwargs,
        )

    def load_model(self):
        self.instance.load_projector(self.model_path)
        return self

    def save_model(self):
        # we have already saved the model in .fit()
        return self


class AlgorithmAsBioAlg(BioAlgorithm, _NonPickableWrapper):
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

    def __init__(self, callable, features_dir, extension=".hdf5", **kwargs):
        super().__init__(callable, **kwargs)
        self.features_dir = features_dir
        self.biometric_reference_dir = os.path.join(
            self.features_dir, "biometric_references"
        )
        self.score_dir = os.path.join(self.features_dir, "scores")
        self.extension = extension

    def _enroll_sample_set(self, sampleset):
        # Enroll
        return self.enroll(sampleset)

    def _score_sample_set(self, sampleset, biometric_references):
        """Given a sampleset for probing, compute the scores and retures a sample set with the scores
        """

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
                subprobe_scores.append(Sample(self.score(ref.data, s.data), parent=ref))

            # Creating one sampleset per probe
            subprobe = SampleSet(subprobe_scores, parent=sampleset)
            subprobe.subprobe_id = subprobe_id

            # Checkpointing score MANDATORY FOR LEGACY
            path = os.path.join(self.score_dir, str(subprobe.path) + ".txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            delayed_scored_sample = save_scores_four_columns(path, subprobe)
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
        if path is None or not os.path.isfile(path):

            # Enrolling
            data = [s.data for s in enroll_features.samples]
            model = self.instance.enroll(data)

            # Checkpointing
            os.makedirs(os.path.dirname(path), exist_ok=True)
            hdf5 = HDF5File(path, "w")
            self.instance.write_model(model, hdf5)

        reader = _get_pickable_method(self.instance.read_model)
        return DelayedSample(functools.partial(reader, path), parent=enroll_features)

    def score(self, model, probe, **kwargs):
        return self.instance.score(model, probe)
