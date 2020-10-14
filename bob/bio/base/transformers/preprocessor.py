#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.preprocessor import Preprocessor


class PreprocessorTransformer(TransformerMixin, BaseEstimator):
    """
    Scikit learn transformer for :py:class:`bob.bio.base.preprocessor.Preprocessor`.

    Parameters
    ----------

      instance: object
         An instance of `bob.bio.base.preprocessor.Preprocessor`


    """

    def __init__(
        self, instance, **kwargs,
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
            return [self.instance(data, annot) for data, annot in zip(X, annotations)]

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None):
        return self
