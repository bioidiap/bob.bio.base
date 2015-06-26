#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.linear
import bob.learn.em

import numpy

from .Algorithm import Algorithm
import logging
logger = logging.getLogger("bob.bio.base")


class PLDA (Algorithm):
  """Tool chain for computing PLDA (over PCA-dimensionality reduced) features

  .. todo:: Add more documentation for the PLDA constructor, i.e., by explaining the parameters

  """

  def __init__(
      self,
      subspace_dimension_of_f, # Size of subspace F
      subspace_dimension_of_g, # Size of subspace G
      subspace_dimension_pca = None,  # if given, perform PCA on data and reduce the PCA subspace to the given dimension
      plda_training_iterations = 200, # Maximum number of iterations for the EM loop
      # TODO: refactor the remaining parameters!
      INIT_SEED = 5489, # seed for initializing
      INIT_F_METHOD = 'BETWEEN_SCATTER',
      INIT_G_METHOD = 'WITHIN_SCATTER',
      INIT_S_METHOD = 'VARIANCE_DATA',
      multiple_probe_scoring = 'joint_likelihood'
  ):

    """Initializes the local (PCA-)PLDA tool chain with the given file selector object"""
    # call base class constructor and register that this class requires training for enrollment
    Algorithm.__init__(
        self,
        requires_enroller_training = True,

        subspace_dimension_of_f = subspace_dimension_of_f, # Size of subspace F
        subspace_dimension_of_g = subspace_dimension_of_g, # Size of subspace G
        subspace_dimension_pca = subspace_dimension_pca,  # if given, perform PCA on data and reduce the PCA subspace to the given dimension
        plda_training_iterations = plda_training_iterations, # Maximum number of iterations for the EM loop
        # TODO: refactor the remaining parameters!
        INIT_SEED = INIT_SEED, # seed for initializing
        INIT_F_METHOD = str(INIT_F_METHOD),
        INIT_G_METHOD = str(INIT_G_METHOD),
        INIT_S_METHOD = str(INIT_S_METHOD),
        multiple_probe_scoring = multiple_probe_scoring,
        multiple_model_scoring = None
    )

    self.subspace_dimension_of_f = subspace_dimension_of_f
    self.subspace_dimension_of_g = subspace_dimension_of_g
    self.subspace_dimension_pca = subspace_dimension_pca
    self.plda_training_iterations = plda_training_iterations
    self.score_set = {'joint_likelihood': 'joint_likelihood', 'average':numpy.average, 'min':min, 'max':max}[multiple_probe_scoring]

    # TODO: refactor
    self.plda_trainer = bob.learn.em.PLDATrainer()
    self.plda_trainer.init_f_method = INIT_F_METHOD
    self.plda_trainer.init_g_method = INIT_G_METHOD
    self.plda_trainer.init_sigma_method = INIT_S_METHOD
    self.rng = bob.core.random.mt19937(INIT_SEED)
    self.pca_machine = None
    self.plda_base = None



  def _train_pca(self, training_set):
    """Trains and returns a LinearMachine that is trained using PCA"""
    data = numpy.vstack([feature for client in training_set for feature in client])

    logger.info("  -> Training LinearMachine using PCA ")
    trainer = bob.learn.linear.PCATrainer()
    machine, eigen_values = trainer.train(data)

    if isinstance(self.subspace_dimension_pca, float):
      cummulated = numpy.cumsum(eigen_values) / numpy.sum(eigen_values)
      for index in range(len(cummulated)):
        if cummulated[index] > self.subspace_dimension_pca:
          self.subspace_dimension_pca = index
          break
      self.subspace_dimension_pca = index

    # limit number of pcs
    logger.info("  -> limiting PCA subspace to %d dimensions", self.subspace_dimension_pca)
    machine.resize(machine.shape[0], self.subspace_dimension_pca)
    return machine

  def _perform_pca_client(self, client):
    """Perform PCA on an array"""
    return numpy.vstack([self.pca_machine(feature) for feature in client])

  def _perform_pca(self, training_set):
    """Perform PCA on data"""
    return [self._perform_pca_client(client) for client in training_set]


  def train_enroller(self, training_features, projector_file):
    """Generates the PLDA base model from a list of arrays (one per identity),
       and a set of training parameters. If PCA is requested, it is trained on the same data.
       Both the trained PLDABase and the PCA machine are written."""


    # train PCA and perform PCA on training data
    if self.subspace_dimension_pca is not None:
      self.pca_machine = self._train_pca(training_features)
      training_features = self._perform_pca(training_features)

    input_dimension = training_features[0].shape[1]
    logger.info("  -> Training PLDA base machine")

    # train machine
    self.plda_base = bob.learn.em.PLDABase(input_dimension, self.subspace_dimension_of_f, self.subspace_dimension_of_g)
    bob.learn.em.train(self.plda_trainer, self.plda_base, training_features, self.plda_training_iterations, self.rng)

    # write machines to file
    proj_hdf5file = bob.io.base.HDF5File(str(projector_file), "w")
    if self.subspace_dimension_pca is not None:
      proj_hdf5file.create_group('/pca')
      proj_hdf5file.cd('/pca')
      self.pca_machine.save(proj_hdf5file)
    proj_hdf5file.create_group('/plda')
    proj_hdf5file.cd('/plda')
    self.plda_base.save(proj_hdf5file)


  def load_enroller(self, projector_file):
    """Reads the PCA projection matrix and the PLDA model from file"""
    # read enroller (PCA and PLDA matrix)
    hdf5 = bob.io.base.HDF5File(projector_file)
    if hdf5.has_group("/pca"):
      hdf5.cd('/pca')
      self.pca_machine = bob.learn.linear.Machine(hdf5)
    hdf5.cd('/plda')
    self.plda_base = bob.learn.em.PLDABase(hdf5)


  def enroll(self, enroll_features):
    """Enrolls the model by computing an average of the given input vectors"""
    plda_machine = bob.learn.em.PLDAMachine(self.plda_base)
    # project features, if enabled
    if self.pca_machine is not None:
      enroll_features = self._perform_pca_client(enroll_features)
    # enroll
    self.plda_trainer.enroll(plda_machine, enroll_features)
    return plda_machine


  def read_model(self, model_file):
    """Reads the model, which in this case is a PLDA-Machine"""
    # read machine and attach base machine
    plda_machine = bob.learn.em.PLDAMachine(bob.io.base.HDF5File(model_file), self.plda_base)
    return plda_machine

  def read_probe(self, probe_file):
    """Reads the probe using :py:func:`bob.bio.base.load`."""
    return bob.bio.base.load(probe_file)


  def score(self, model, probe):
    """Computes the PLDA score for the given model and probe"""
    return self.score_for_multiple_probes(model, [probe])


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files.
    In this base class implementation, it computes the scores for each probe file using the 'score' method,
    and fuses the scores using the fusion method specified in the constructor of this class."""
    if self.pca_machine is not None:
      probes = [self.pca_machine(probe) for probe in probes]
    # forward
    if self.score_set == 'joint_likelihood':
      return model.log_likelihood_ratio(numpy.vstack(probes))
    else:
      return self.score_set([model.log_likelihood_ratio(probe) for probe in probes])

  # re-define unused functions, just so that they do not get documented
  def train_projector(*args,**kwargs): raise NotImplementedError()
  def load_projector(*args,**kwargs): pass
  def project(*args,**kwargs): raise NotImplementedError()
  def write_feature(*args,**kwargs): raise NotImplementedError()
  def read_feature(*args,**kwargs): raise NotImplementedError()
