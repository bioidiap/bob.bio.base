#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.algorithm import Algorithm
from bob.pipelines.utils import is_picklable
from . import split_X_by_y
import os

class AlgorithmTransformer(TransformerMixin, BaseEstimator):
    """Class that wraps :any:`bob.bio.base.algorithm.Algoritm`

    :any:`LegacyAlgorithmrMixin.fit` maps to :any:`bob.bio.base.algorithm.Algoritm.train_projector`

    :any:`LegacyAlgorithmrMixin.transform` maps :any:`bob.bio.base.algorithm.Algoritm.project`

    Example
    -------

        Wrapping LDA algorithm with functtools
        >>> from bob.bio.base.pipelines.vanilla_biometrics.legacy import LegacyAlgorithmAsTransformer
        >>> from bob.bio.base.algorithm import LDA
        >>> import functools
        >>> transformer = LegacyAlgorithmAsTransformer(functools.partial(LDA, use_pinv=True, pca_subspace_dimension=0.90))


    Parameters
    ----------
      callable: ``collections.callable``
         Callable function that instantiates the bob.bio.base.algorithm.Algorithm

    """

    def __init__(
        self, callable, projector_file=None, **kwargs,
    ):

        if not isinstance(callable, Algorithm):
            raise ValueError(
                "`callable` should be an instance of `bob.bio.base.extractor.Algorithm`"
            )

        if callable.requires_training and (
            projector_file is None or projector_file == ""
        ):
            raise ValueError(
                f"`projector_file` needs to be set if extractor {callable} requires training"
            )

        if not is_picklable(callable):
            raise ValueError(f"{callable} needs to be picklable")

        self.callable = callable
        self.projector_file = projector_file
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        if not self.callable.requires_training:
            return self        
        training_data = X        
        if self.callable.split_training_features_by_client:
            training_data = split_X_by_y(X, y)

        os.makedirs(os.path.dirname(self.projector_file), exist_ok=True)
        self.callable.train_projector(training_data, self.projector_file)
        return self

    def transform(self, X, metadata=None):        
        if metadata is None:
            return [self.callable.project(data) for data in X]
        else:
            return [
                self.callable.project(data, metadata)
                for data, metadata in zip(X, metadata)
            ]

    def _more_tags(self):
        if self.callable.requires_training:
            return {"stateless": False, "requires_fit": True}
        else:
            return {"stateless": True, "requires_fit": False}
