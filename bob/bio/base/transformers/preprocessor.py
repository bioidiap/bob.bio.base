#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.preprocessor import Preprocessor

class PreprocessorTransformer(TransformerMixin, BaseEstimator):
    """
    Scikit learn transformer for :any:`bob.bio.base.preprocessor.Preprocessor`.

    Parameters
    ----------

      callable: ``collections.Callable``
         Instance of `bob.bio.base.preprocessor.Preprocessor`


    """

    def __init__(
        self,
        callable,
        **kwargs,
    ):

        if not isinstance(callable, Preprocessor):
            raise ValueError("`callable` should be an instance of `bob.bio.base.preprocessor.Preprocessor`")

        self.callable = callable
        super().__init__(**kwargs)

    def transform(self, X, annotations=None):
        if annotations is None:
            return [self.callable(data) for data in X]
        else:
            return [self.callable(data, annot) for data, annot in zip(X, annotations)]

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
