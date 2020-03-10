#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.pipelines.processor import CheckpointMixin, SampleMixin
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy


class Linearize(TransformerMixin):
    """Extracts features by simply concatenating all elements of the data into one long vector.

    If a ``dtype`` is specified in the contructor, it is assured that the resulting
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        """__call__(data) -> data

        Takes data of arbitrary dimensions and linearizes it into a 1D vector; enforcing the data type, if desired.

        Parameters:
        -----------

        data : :py:class:`numpy.ndarray`
          The preprocessed data to be transformed into one vector.

        Returns:
        --------

        data : 1D :py:class:`numpy.ndarray`
          The extracted feature vector, of the desired ``dtype`` (if specified).
        """

        X = check_array(X, allow_nd=True)

        if X.ndim == 2:
            return numpy.reshape(X, X.size)
        else:
            # Reshaping n-dimensional arrays assuming that the
            # first axis corresponds to the number of samples
            return numpy.reshape(X, (X.shape[0], numpy.prod(X.shape[1:])))


class SampleLinearize(SampleMixin, Linearize):
    pass


class CheckpointSampleLinearize(CheckpointMixin, SampleMixin, Linearize):
    pass
