#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Fri Oct 26 17:05:40 CEST 2012

from .Extractor import Extractor
import numpy


class Linearize(Extractor):
    """Extracts features by simply concatenating all elements of the data into one long vector.

    If a ``dtype`` is specified in the contructor, it is assured that the resulting
    """

    def __init__(self, dtype=None):
        """If the ``dtype`` parameter is given, it specifies the data type that is enforced for the features."""
        super(Linearize, self).__init__(dtype=dtype)
        self.dtype = dtype

    def __call__(self, data):
        """__call__(data) -> data

        Takes data of arbitrary dimensions and linearizes it into a 1D vector; enforcing the data type, if desired.

        **Parameters:**

        data : :py:class:`numpy.ndarray`
          The preprocessed data to be transformed into one vector.

        **Returns:**

        data : 1D :py:class:`numpy.ndarray`
          The extracted feature vector, of the desired ``dtype`` (if specified).
        """        
        assert isinstance(data, numpy.ndarray)

        linear = numpy.reshape(data, data.size)
        if self.dtype is not None:
            linear = linear.astype(self.dtype)
        return linear

    # re-define unused functions, just so that they do not get documented
    def train(*args, **kwargs): raise NotImplementedError()

    def load(*args, **kwargs): pass
