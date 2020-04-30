#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.extractor import Extractor
from . import split_X_by_y

class ExtractorTransformer(TransformerMixin, BaseEstimator):
    """
    Scikit learn transformer for :any:`bob.bio.base.extractor.Extractor`.

    Parameters
    ----------

      callable: ``collections.Callable``
         Instance of `bob.bio.base.extractor.Extractor`

      model_path: ``str``
         Model path in case :any:`bob.bio.base.extractor.Extractor.requires_training` is equals to true

    """

    def __init__(
        self, callable, model_path=None, **kwargs,
    ):

        if not isinstance(callable, Extractor):
            raise ValueError(
                "`callable` should be an instance of `bob.bio.base.extractor.Extractor`"
            )

        if callable.requires_training and (model_path is None or model_path==""):
            raise ValueError(
                f"`model_path` needs to be set if extractor {callable} requires training"
            )

        self.callable = callable
        self.model_path = model_path
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        if not self.callable.requires_training:
            return self

        training_data = X
        if self.callable.split_training_data_by_client:
            training_data = split_X_by_y(X, y)

        self.callable.train(training_data, self.model_path)
        return self

    def transform(self, X, metadata=None):
        if metadata is None:
            return [self.callable(data) for data in X]
        else:
            return [
                self.callable(data, metadata) for data, metadata in zip(X, metadata)
            ]

    def _more_tags(self):
        return {
            "stateless": not self.callable.requires_training,
            "requires_fit": self.callable.requires_training,
        }
