#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.algorithm import Algorithm
from bob.pipelines.utils import is_picklable
from . import split_X_by_y
import os


class AlgorithmTransformer(TransformerMixin, BaseEstimator):
    """Class that wraps :py:class:`bob.bio.base.algorithm.Algorithm`

    :any:`LegacyAlgorithmMixin.fit` maps to :py:meth:`bob.bio.base.algorithm.Algorithm.train_projector`

    :any:`LegacyAlgorithmMixin.transform` maps :py:meth:`bob.bio.base.algorithm.Algorithm.project`

    Example
    -------

        Wrapping LDA algorithm with functools
        >>> from bob.bio.base.pipelines.vanilla_biometrics import AlgorithmTransformer
        >>> from bob.bio.base.algorithm import LDA
        >>> transformer = AlgorithmTransformer(LDA(use_pinv=True, pca_subspace_dimension=0.90)


    Parameters
    ----------
      instance: object
         An instance of bob.bio.base.algorithm.Algorithm

    """

    def __init__(
        self, instance, projector_file=None, **kwargs,
    ):

        if not isinstance(instance, Algorithm):
            raise ValueError(
                "`instance` should be an instance of `bob.bio.base.extractor.Algorithm`"
            )

        if instance.requires_projector_training and (
            projector_file is None or projector_file == ""
        ):
            raise ValueError(
                f"`projector_file` needs to be set if extractor {instance} requires training"
            )

        if not is_picklable(instance):
            raise ValueError(f"{instance} needs to be picklable")

        self.instance = instance
        self.projector_file = projector_file
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        if not self.instance.requires_projector_training:
            return self
        training_data = X
        if self.instance.split_training_features_by_client:
            training_data = split_X_by_y(X, y)

        os.makedirs(os.path.dirname(self.projector_file), exist_ok=True)
        self.instance.train_projector(training_data, self.projector_file)
        return self

    def transform(self, X, metadata=None):
        if metadata is None:
            return [self.instance.project(data) for data in X]
        else:
            return [
                self.instance.project(data, metadata)
                for data, metadata in zip(X, metadata)
            ]

    def _more_tags(self):
        return {
            "stateless": not self.instance.requires_projector_training,
            "requires_fit": self.instance.requires_projector_training,
        }
