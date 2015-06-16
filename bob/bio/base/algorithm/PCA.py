#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.learn.linear
import bob.io.base

import numpy
import scipy.spatial

from .Algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.base")

class PCA (Algorithm):
  """Performs PCA on the given data"""

  def __init__(
      self,
      subspace_dimension,  # if int, number of subspace dimensions; if float, percentage of variance to keep
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      **kwargs  # parameters directly sent to the base class
  ):

    """Initializes the PCA Algorithm with the given setup"""
    # call base class constructor and register that the tool performs a projection
    Algorithm.__init__(
        self,
        performs_projection = True,

        subspace_dimension = subspace_dimension,
        distance_function = str(distance_function),
        is_distance_function = is_distance_function,
        uses_variances = uses_variances,

        **kwargs
    )

    self.subspace_dim = subspace_dimension
    self.machine = None
    self.distance_function = distance_function
    self.factor = -1. if is_distance_function else 1.
    self.uses_variances = uses_variances


  def _check_feature(self, feature, projected=False):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray) or feature.ndim != 1 or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")
    index = 1 if projected else 0
    if self.machine is not None and feature.shape[0] != self.machine.shape[index]:
      raise ValueError("The given feature is expected to have %d elements, but it has %d" % (self.machine.shape[index], feature.shape[0]))


  def train_projector(self, training_features, projector_file):
    """Generates the PCA covariance matrix"""
    # Assure that all data are 1D
    [self._check_feature(feature) for feature in training_features]

    # Initializes the data
    data = numpy.vstack(training_features)
    logger.info("  -> Training LinearMachine using PCA")
    t = bob.learn.linear.PCATrainer()
    self.machine, self.variances = t.train(data)
    # For re-shaping, we need to copy...
    self.variances = self.variances.copy()

    # compute variance percentage, if desired
    if isinstance(self.subspace_dim, float):
      cummulated = numpy.cumsum(self.variances) / numpy.sum(self.variances)
      for index in range(len(cummulated)):
        if cummulated[index] > self.subspace_dim:
          self.subspace_dim = index
          break
      self.subspace_dim = index
    logger.info("    ... Keeping %d PCA dimensions", self.subspace_dim)
    # re-shape machine
    self.machine.resize(self.machine.shape[0], self.subspace_dim)
    self.variances.resize(self.subspace_dim)

    f = bob.io.base.HDF5File(projector_file, "w")
    f.set("Eigenvalues", self.variances)
    f.create_group("Machine")
    f.cd("/Machine")
    self.machine.save(f)


  def load_projector(self, projector_file):
    """Reads the PCA projection matrix from file"""
    # read PCA projector
    f = bob.io.base.HDF5File(projector_file)
    self.variances = f.read("Eigenvalues")
    f.cd("/Machine")
    self.machine = bob.learn.linear.Machine(f)


  def project(self, feature):
    """Projects the data using the stored covariance matrix"""
    self._check_feature(feature)
    # Projects the data
    return self.machine(feature)


  def enroll(self, enroll_features):
    """Enrolls the model by storing all given input vectors"""
    assert len(enroll_features)
    [self._check_feature(feature, True) for feature in enroll_features]
    # just store all the features
    return numpy.vstack(enroll_features)


  def score(self, model, probe):
    """Computes the distance of the model to the probe using the distance function"""
    self._check_feature(probe, True)
    # return the negative distance (as a similarity measure)
    if len(model.shape) == 2:
      # we have multiple models, so we use the multiple model scoring
      return self.score_for_multiple_models(model, probe)
    elif self.uses_variances:
      # single model, single probe (multiple probes have already been handled)
      return self.factor * self.distance_function(model, probe, self.variances)
    else:
      # single model, single probe (multiple probes have already been handled)
      return self.factor * self.distance_function(model, probe)
