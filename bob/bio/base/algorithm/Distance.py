#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Gunther <siebenkopf@googlemail.com>

import numpy
import scipy.spatial

from .Algorithm import Algorithm
from .. import utils

import logging
logger = logging.getLogger("bob.bio.base")

class Distance (Algorithm):
  """This class defines a simple distance measure between two features.
  Independent of the actual shape, each feature vector is treated as a one-dimensional vector, and the specified distance function is used to compute the distance between the two features.
  If the given ``distance_function`` actually computes a distance, we simply return its negative value (as all :py:class:`Algorithm`'s are supposed to return similarity values).
  If the ``distance_function`` computes similarities, the similarity value is returned unaltered.

  **Parameters:**

  ``distance_function`` : callable
    A function taking two 1D arrays and returning a ``float``

  ``is_distance_function`` : bool
    Set this flag to ``False`` if the given ``distance_function`` computes a similarity value (i.e., higher values are better)

  ``kwargs`` : ``key=value`` pairs
    A list of keyword arguments directly passed to the :py:class:`Algorithm` base class constructor.
  """

  def __init__(
      self,
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      **kwargs  # parameters directly sent to the base class
  ):

    # call base class constructor and register that the algorithm performs a projection
    Algorithm.__init__(
        self,
        distance_function = str(distance_function),
        is_distance_function = is_distance_function,

        **kwargs
    )

    self.distance_function = distance_function
    self.factor = -1. if is_distance_function else 1.

  def _check_feature(self, feature):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray):
      raise ValueError("The given feature should be of type numpy.ndarray")

  def enroll(self, enroll_features):
    """enroll(enroll_features) -> model

    Enrolls the model by storing all given input vectors.

    **Parameters:**

    ``enroll_features`` : [:py:class:`numpy.ndarray`]
      The list of projected features to enroll the model from.

    **Returns:**

    ``model`` : 2D :py:class:`numpy.ndarray`
      The enrolled model.
    """
    assert len(enroll_features)
    [self._check_feature(feature) for feature in enroll_features]
    # just store all the features
    return numpy.vstack([f.flatten() for f in enroll_features])


  def read_probe(self, probe_file):
    """read_probe(probe_file) -> probe

    Reads the probe feature from the given HDF5 file.

    **Parameters:**

    probe_file : str or :py:class:`bob.io.base.HDF5File`
      The file (open for reading) or the name of an existing file to read from.

    **Returns:**

    probe : object
      The probe.
    """
    return utils.load(probe_file)


  def score(self, model, probe):
    """score(model, probe) -> float

    Computes the distance of the model to the probe using the distance function specified in the constructor.

    **Parameters:**

    ``model`` : 2D :py:class:`numpy.ndarray`
      The model storing all enrollment features

    ``probe`` : :py:class:`numpy.ndarray`
      The probe feature vector

    **Returns:**

    ``score`` : float
      A similarity value between ``model`` and ``probe``
    """
    self._check_feature(probe)
    probe = probe.flatten()
    # return the negative distance (as a similarity measure)
    if model.ndim == 2:
      # we have multiple models, so we use the multiple model scoring
      return self.score_for_multiple_models(model, probe)
    else:
      # single model, single probe (multiple probes have already been handled)
      return self.factor * self.distance_function(model, probe)

  # re-define unused functions, just so that they do not get documented
  def train_projector(*args,**kwargs): raise NotImplementedError()
  def load_projector(*args,**kwargs): pass
  def project(*args,**kwargs): raise NotImplementedError()
  def write_feature(*args,**kwargs): raise NotImplementedError()
  def read_feature(*args,**kwargs): raise NotImplementedError()
  def train_enroller(*args,**kwargs): raise NotImplementedError()
  def load_enroller(*args,**kwargs): pass
