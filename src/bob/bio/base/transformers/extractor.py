#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.base.extractor import Extractor

from . import split_X_by_y


class ExtractorTransformer(TransformerMixin, BaseEstimator):
    """Scikit learn transformer for :py:class:`bob.bio.base.extractor.Extractor`.

    Parameters
    ----------
    instance: object
        An instance of :py:class:`bob.bio.base.extractor.Extractor`

    model_path: ``str``
        Model path in case ``instance.requires_training`` is equal to ``True``.
    """

    def __init__(
        self,
        instance,
        model_path=None,
        **kwargs,
    ):
        if not isinstance(instance, Extractor):
            raise ValueError(
                "`instance` should be an instance of `bob.bio.base.extractor.Extractor`"
            )

        if instance.requires_training and (
            model_path is None or model_path == ""
        ):
            raise ValueError(
                f"`model_path` needs to be set if extractor {instance} requires training"
            )

        self.instance = instance
        self.model_path = model_path
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        if not self.instance.requires_training:
            return self

        training_data = X
        if self.instance.split_training_data_by_client:
            training_data = split_X_by_y(X, y)

        self.instance.train(training_data, self.model_path)
        return self

    def transform(self, X, metadata=None):
        if metadata is None:
            return [self.instance(data) for data in X]
        else:
            return [
                self.instance(data, metadata)
                for data, metadata in zip(X, metadata)
            ]

    def _more_tags(self):
        return {
            "requires_fit": self.instance.requires_training,
            "bob_features_save_fn": self.instance.write_feature,
            "bob_features_load_fn": self.instance.read_feature,
        }
