#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.base.preprocessor import Preprocessor


class PreprocessorTransformer(TransformerMixin, BaseEstimator):
    """Scikit learn transformer for :py:class:`bob.bio.base.preprocessor.Preprocessor`.

    Parameters
    ----------
    instance: object
        An instance of `bob.bio.base.preprocessor.Preprocessor`
    """

    def __init__(
        self,
        instance,
        **kwargs,
    ):

        if not isinstance(instance, Preprocessor):
            raise ValueError(
                "`instance` should be an instance of `bob.bio.base.preprocessor.Preprocessor`"
            )

        self.instance = instance
        super().__init__(**kwargs)

    def transform(self, X, annotations=None):
        if annotations is None:
            return [self.instance(data) for data in X]
        else:
            return [
                self.instance(data, annot)
                for data, annot in zip(X, annotations)
            ]

    def _more_tags(self):
        return {
            "requires_fit": False,
            "bob_features_save_fn": self.instance.write_data,
            "bob_features_load_fn": self.instance.read_data,
        }

    def fit(self, X, y=None):
        return self
