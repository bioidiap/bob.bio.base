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
  """Performs a principal component analysis (PCA) on the given data.

  This algorithm computes a PCA projection (:py:class:`bob.learn.linear.PCATrainer`) on the given training features, projects the features to eigenspace and computes the distance of two projected features in eigenspace.
  For example, the eigenface algorithm as proposed by [TP91]_ can be run with this class.

  **Parameters:**

  subspace_dimension : int or float
    If specified as ``int``, defines the number of eigenvectors used in the PCA projection matrix.
    If specified as ``float`` (between 0 and 1), the number of eigenvectors is calculated such that the given percentage of variance is kept.

  distance_function : function
    A function taking two parameters and returns a float.
    If ``uses_variances`` is set to ``True``, the function is provided with a third parameter, which is the vector of variances (aka. eigenvalues).

  is_distance_function : bool
    Set this flag to ``False`` if the given ``distance_function`` computes a similarity value (i.e., higher values are better)

  use_variances : bool
    If set to ``True``, the ``distance_function`` is provided with a third argument, which is the vector of variances (aka. eigenvalues).

  kwargs : ``key=value`` pairs
    A list of keyword arguments directly passed to the :py:class:`Algorithm` base class constructor.
  """

  def __init__(
      self,
      subspace_dimension,  # if int, number of subspace dimensions; if float, percentage of variance to keep
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      **kwargs  # parameters directly sent to the base class
  ):

    # call base class constructor and register that the algorithm performs a projection
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
    """Generates the PCA covariance matrix and writes it into the given projector_file.

    **Parameters:**

    training_features : [1D :py:class:`numpy.ndarray`]
      A list of 1D training arrays (vectors) to train the PCA projection matrix with.

    projector_file : str
      A writable file, into which the PCA projection matrix (as a :py:class:`bob.learn.linear.Machine`) and the eigenvalues will be written.
    """
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
    """Reads the PCA projection matrix and the eigenvalues from file.

    **Parameters:**

    projector_file : str
      An existing file, from which the PCA projection matrix and the eigenvalues are read.
    """
    # read PCA projector
    f = bob.io.base.HDF5File(projector_file)
    self.variances = f.read("Eigenvalues")
    f.cd("/Machine")
    self.machine = bob.learn.linear.Machine(f)


  def project(self, feature):
    """project(feature) -> projected

    Projects the given feature into eigenspace.

    **Parameters:**

    feature : 1D :py:class:`numpy.ndarray`
      The 1D feature to be projected.

    **Returns:**

    projected : 1D :py:class:`numpy.ndarray`
      The ``feature`` projected into eigenspace.
    """
    self._check_feature(feature)
    # Projects the data
    return self.machine(feature)


  def enroll(self, enroll_features):
    """enroll(enroll_features) -> model

    Enrolls the model by storing all given input vectors.

    **Parameters:**

    enroll_features : [1D :py:class:`numpy.ndarray`]
      The list of projected features to enroll the model from.

    **Returns:**

    model : 2D :py:class:`numpy.ndarray`
      The enrolled model.
    """
    assert len(enroll_features)
    [self._check_feature(feature, True) for feature in enroll_features]
    # just store all the features
    return numpy.vstack(enroll_features)


  def score(self, model, probe):
    """score(model, probe) -> float

    Computes the distance of the model to the probe using the distance function specified in the constructor.

    **Parameters:**

    model : 2D :py:class:`numpy.ndarray`
      The model storing all enrollment features.

    probe : 1D :py:class:`numpy.ndarray`
      The probe feature vector in eigenspace.

    **Returns:**

    score : float
      A similarity value between ``model`` and ``probe``
    """
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

  # re-define unused functions, just so that they do not get documented
  def train_enroller(*args,**kwargs): raise NotImplementedError()
  def load_enroller(*args,**kwargs): pass
