#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.io.base
import bob.learn.linear

import numpy
import scipy.spatial

from .Algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.base")

class LDA (Algorithm):
  """Computes a linear discriminant analysis (LDA) on the given data, possibly after computing a principal component analysis (PCA).

  This algorithm computes a LDA projection (:py:class:`bob.learn.linear.FisherLDATrainer`) on the given training features, projects the features to Fisher space and computes the distance of two projected features in Fisher space.
  For example, the Fisher faces algorithm as proposed by [ZKC+98]_ can be run with this class.


  Additionally, a PCA projection matrix can be computed beforehand, to reduce the dimensionality of the input vectors.
  In that case, the finally stored projection matrix is the combination of the PCA and LDA projection.

  **Parameters:**

  lda_subspace_dimension : int or ``None``
    If specified, the LDA subspace will be truncated to the given number of dimensions.
    By default (``None``) it is limited to the number of classes in the training set - 1.

  pca_subspace_dimentsion : int or float or ``None``
    If specified, a combined PCA + LDA projection matrix will be computed.
    If specified as ``int``, defines the number of eigenvectors used in the PCA projection matrix.
    If specified as ``float`` (between 0 and 1), the number of eigenvectors is calculated such that the given percentage of variance is kept.

  use_pinv : bool
    Use the Pseudo-inverse to compute the LDA projection matrix?
    Sometimes, the training fails because it is impossible to invert the covariance matrix.
    In these cases, you might want to set ``use_pinv`` to ``True``, which solves this process, but slows down the processing noticeably.

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
      lda_subspace_dimension = None, # if set, the LDA subspace will be truncated to the given number of dimensions; by default it is limited to the number of classes in the training set
      pca_subspace_dimension = None, # if set, a PCA subspace truncation is performed before applying LDA; might be integral or float
      use_pinv = False,
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      **kwargs  # parameters directly sent to the base class
  ):

    # call base class constructor and register that the LDA tool performs projection and need the training features split by client
    Algorithm.__init__(
        self,
        performs_projection = True,
        split_training_features_by_client = True,

        lda_subspace_dimension = lda_subspace_dimension,
        pca_subspace_dimension = pca_subspace_dimension,
        use_pinv = use_pinv,
        distance_function = str(distance_function),
        is_distance_function = is_distance_function,
        uses_variances = uses_variances,

        **kwargs
    )

    # copy information
    self.pca_subspace = pca_subspace_dimension
    self.lda_subspace = lda_subspace_dimension
    if self.pca_subspace is not None and isinstance(self.pca_subspace, int) and self.lda_subspace and self.pca_subspace < self.lda_subspace:
      raise ValueError("The LDA subspace is larger than the PCA subspace size. This won't work properly. Please check your setup!")
    self.use_pinv = use_pinv

    self.machine = None
    self.distance_function = distance_function
    self.factor = -1 if is_distance_function else 1.
    self.uses_variances = uses_variances


  def _check_feature(self, feature, projected=False):
    """Checks that the features are appropriate."""
    if not isinstance(feature, numpy.ndarray) or feature.ndim != 1 or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")
    index = 1 if projected else 0
    if self.machine is not None and feature.shape[0] != self.machine.shape[index]:
      raise ValueError("The given feature is expected to have %d elements, but it has %d" % (self.machine.shape[index], feature.shape[0]))


  def _arrange_data(self, training_files):
    """Arranges the data to train the LDA projection matrix."""
    data = []
    for client_files in training_files:
      # at least two files per client are required!
      if len(client_files) < 2:
        logger.warn("Skipping one client since the number of client files is only %d", len(client_files))
        continue
      data.append(numpy.vstack([feature.flatten() for feature in client_files]))

    # Returns the list of lists of arrays
    return data


  def _train_pca(self, training_set):
    """Trains and returns a LinearMachine that is trained using PCA"""
    data_list = [feature for client in training_set for feature in client]
    data = numpy.vstack(data_list)

    logger.info("  -> Training Linear Machine using PCA")
    t = bob.learn.linear.PCATrainer()
    machine, eigen_values = t.train(data)

    if isinstance(self.pca_subspace, float):
      cummulated = numpy.cumsum(eigen_values) / numpy.sum(eigen_values)
      for index in range(len(cummulated)):
        if cummulated[index] > self.pca_subspace:
          self.pca_subspace = index
          break
      self.pca_subspace = index

    if self.lda_subspace is not None and self.pca_subspace <= self.lda_subspace:
      logger.warn("  ... Extending the PCA subspace dimension from %d to %d", self.pca_subspace, self.lda_subspace + 1)
      self.pca_subspace = self.lda_subspace + 1
    else:
      logger.info("  ... Limiting PCA subspace to %d dimensions", self.pca_subspace)

    # limit number of pcs
    machine.resize(machine.shape[0], self.pca_subspace)
    return machine


  def _perform_pca(self, machine, training_set):
    """Perform PCA on data of the training set"""
    return [numpy.vstack([machine(feature) for feature in client_features]) for client_features in training_set]


  def train_projector(self, training_features, projector_file):
    """Generates the LDA or PCA+LDA projection matrix from the given features (that are sorted by identity).

    **Parameters:**

    training_features : [[1D :py:class:`numpy.ndarray`]]
      A list of lists of 1D training arrays (vectors) to train the LDA projection matrix with.
      Each sub-list contains the features of one client.

    projector_file : str
      A writable file, into which the LDA or PCA+LDA projection matrix (as a :py:class:`bob.learn.linear.Machine`) and the eigenvalues will be written.
    """
    # check data
    [self._check_feature(feature) for client_features in training_features for feature in client_features]

    # arrange LDA training data
    data = self._arrange_data(training_features)

    # train PCA of wanted
    if self.pca_subspace:
      # train on all training features
      pca_machine = self._train_pca(training_features)
      # project only the features that are used for training
      logger.info("  -> Projecting training data to PCA subspace")
      data = self._perform_pca(pca_machine, data)

    logger.info("  -> Training Linear Machine using LDA")
    trainer = bob.learn.linear.FisherLDATrainer(use_pinv = self.use_pinv, strip_to_rank = (self.lda_subspace is None))
    self.machine, self.variances = trainer.train(data)
    if self.lda_subspace is not None:
      self.machine.resize(self.machine.shape[0], self.lda_subspace)
      self.variances = self.variances.copy()
      self.variances.resize(self.lda_subspace)

    if self.pca_subspace is not None:
      # compute combined PCA/LDA projection matrix
      combined_matrix = numpy.dot(pca_machine.weights, self.machine.weights)
      # set new weight matrix (and new mean vector) of novel machine
      self.machine = bob.learn.linear.Machine(combined_matrix)
      self.machine.input_subtract = pca_machine.input_subtract

    hdf5 = bob.io.base.HDF5File(projector_file, "w")
    hdf5.set("Eigenvalues", self.variances)
    hdf5.create_group("/Machine")
    hdf5.cd("/Machine")
    self.machine.save(hdf5)


  def load_projector(self, projector_file):
    """Reads the projection matrix and the eigenvalues from file.

    **Parameters:**

    projector_file : str
      An existing file, from which the PCA or PCA+LDA projection matrix and the eigenvalues are read.
    """
    # read LDA projector
    hdf5 = bob.io.base.HDF5File(projector_file)
    self.variances = hdf5.read("Eigenvalues")
    hdf5.cd("/Machine")
    self.machine = bob.learn.linear.Machine(hdf5)


  def project(self, feature):
    """project(feature) -> projected

    Projects the given feature into Fisher space.

    **Parameters:**

    feature : 1D :py:class:`numpy.ndarray`
      The 1D feature to be projected.

    **Returns:**

    projected : 1D :py:class:`numpy.ndarray`
      The ``feature`` projected into Fisher space.
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
      The probe feature vector in Fisher space.

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
