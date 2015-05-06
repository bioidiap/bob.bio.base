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
  """Tool for computing linear discriminant analysis (so-called Fisher faces)"""

  def __init__(
      self,
      lda_subspace_dimension = 0, # if set, the LDA subspace will be truncated to the given number of dimensions; by default it is limited to the number of classes in the training set
      pca_subspace_dimension = None, # if set, a PCA subspace truncation is performed before applying LDA; might be integral or float
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      **kwargs  # parameters directly sent to the base class
  ):
    """Initializes the LDA tool with the given configuration"""

    # call base class constructor and register that the LDA tool performs projection and need the training features split by client
    Algorithm.__init__(
        self,
        performs_projection = True,
        split_training_features_by_client = True,

        lda_subspace_dimension = lda_subspace_dimension,
        pca_subspace_dimension = pca_subspace_dimension,
        distance_function = str(distance_function),
        is_distance_function = is_distance_function,
        uses_variances = uses_variances,

        **kwargs
    )

    # copy information
    self.pca_subspace = pca_subspace_dimension
    self.lda_subspace = lda_subspace_dimension
    if self.pca_subspace and isinstance(self.pca_subspace, int) and self.lda_subspace and self.pca_subspace < self.lda_subspace:
      raise ValueError("The LDA subspace is larger than the PCA subspace size. This won't work properly. Please check your setup!")

    self.machine = None
    self.distance_function = distance_function
    self.factor = -1 if is_distance_function else 1.
    self.uses_variances = uses_variances


  def _check_feature(self, feature, projected=False):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray) or len(feature.shape) != 1 or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")
    index = 1 if projected else 0
    if self.machine is not None and feature.shape[0] != self.machine.shape[index]:
      raise ValueError("The given feature is expected to have %d elements, but it has %d" % (self.machine.shape[index], feature.shape[0]))


  def _arrange_data(self, training_files):
    """Arranges the data to train the LDA projection matrix"""
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

    if self.lda_subspace and self.pca_subspace <= self.lda_subspace:
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
    """Generates the LDA projection matrix from the given features (that are sorted by identity)"""
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
    trainer = bob.learn.linear.FisherLDATrainer(strip_to_rank = (self.lda_subspace == 0))
    self.machine, self.variances = trainer.train(data)
    if self.lda_subspace:
      self.machine.resize(self.machine.shape[0], self.lda_subspace)
      self.variances = self.variances.copy()
      self.variances.resize(self.lda_subspace)

    if self.pca_subspace:
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
    """Reads the LDA projection matrix from file"""
    # read LDA projector
    hdf5 = bob.io.base.HDF5File(projector_file)
    self.variances = hdf5.read("Eigenvalues")
    hdf5.cd("/Machine")
    self.machine = bob.learn.linear.Machine(hdf5)


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
