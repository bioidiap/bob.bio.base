#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Mixins to handle legacy components
"""

from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array


class LegacyProcessorMixin(TransformerMixin):
    """Class that wraps :py:class:`bob.bio.base.preprocessor.Preprocessor` and
    :py:class:`bob.bio.base.extractor.Extractors`


    Example
    -------

        Wrapping preprocessor with functtools
        >>> from bob.bio.base.mixins.legacy import LegacyProcessorMixin
        >>> from bob.bio.face.preprocessor import FaceCrop
        >>> import functools
        >>> transformer = LegacyProcessorMixin(functools.partial(FaceCrop, cropped_image_size=(10,10)))

    Example
    -------
        Wrapping extractor 
        >>> from bob.bio.base.mixins.legacy import LegacyProcessorMixin
        >>> from bob.bio.face.extractor import Linearize
        >>> transformer = LegacyProcessorMixin(Linearize)


    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the scikit estimator

    """

    def __init__(self, callable=None):
        self.callable = callable
        self.instance = None

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):

        X = check_array(X, allow_nd=True)

        # Instantiates and do the "real" transform
        if self.instance is None:
            self.instance = self.callable()
        return [self.instance(x) for x in X]
