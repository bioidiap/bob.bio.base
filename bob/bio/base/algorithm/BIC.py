#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.io.base
import bob.learn.linear

import numpy
import math

from .Algorithm import Algorithm
from .. import utils

import logging
logger = logging.getLogger("bob.bio.base")

class BIC (Algorithm):
  """Computes the Intrapersonal/Extrapersonal classifier using a generic feature type and feature comparison function"""

  def __init__(
      self,
      comparison_function, # the function to be used to compare two features; this highly depends on the type of features that are used
      maximum_training_pair_count = None,  # if set, limit the number of training pairs to the given number in a non-random manner
      subspace_dimensions = None, # if set as a pair (intra_dim, extra_dim), PCA subspace truncation for the two classes is performed
      uses_dffs = False, # use the distance from feature space; only valid when PCA truncation is enabled; WARNING: uses this flag with care
      read_function = utils.load,
      write_function = utils.save,
      **kwargs # parameters directly sent to the base class
  ):

    # call base class function and register that this tool requires training for the enrollment
    Algorithm.__init__(
        self,
        requires_enroller_training = True,

        comparison_function = str(comparison_function),
        maximum_training_pair_count = maximum_training_pair_count,
        subspace_dimensions = subspace_dimensions,
        uses_dffs = uses_dffs,
        read_function=str(read_function),
        write_function=str(write_function),

        **kwargs
    )

    # set up the BIC tool
    self.comparison_function = comparison_function
    self.read_function = read_function
    self.write_function = write_function
    self.maximum_pair_count = maximum_training_pair_count
    self.use_dffs = uses_dffs
    if subspace_dimensions is not None:
      self.M_I = subspace_dimensions[0]
      self.M_E = subspace_dimensions[1]
      self.bic_machine = bob.learn.linear.BICMachine(self.use_dffs)
    else:
      self.bic_machine = bob.learn.linear.BICMachine(False)
      self.M_I = None
      self.M_E = None


  def _sqr(self, x):
    return x*x


  def _trainset_for(self, pairs):
    """Computes the array containing the comparison results for the given set of image pairs."""
    return numpy.vstack([self.comparison_function(f1, f2) for (f1, f2) in pairs])


  def train_enroller(self, train_features, enroller_file):
    """Trains the IEC Tool, i.e., computes intrapersonal and extrapersonal subspaces"""

    # compute intrapersonal and extrapersonal pairs
    logger.info("  -> Computing pairs")
    intra_pairs, extra_pairs = bob.learn.linear.bic_intra_extra_pairs(train_features)
    # limit pairs, if desired
    if self.maximum_pair_count is not None:
      if len(intra_pairs) > self.maximum_pair_count:
        logger.info("  -> Limiting intrapersonal pairs from %d to %d" %(len(intra_pairs), self.maximum_pair_count))
        intra_pairs = utils.selected_elements(intra_pairs, self.maximum_pair_count)
      if len(extra_pairs) > self.maximum_pair_count:
        logger.info("  -> Limiting extrapersonal pairs from %d to %d" %(len(extra_pairs), self.maximum_pair_count))
        extra_pairs = utils.selected_elements(extra_pairs, self.maximum_pair_count)


    # train the BIC Machine with these pairs
    logger.info("  -> Computing %d intrapersonal results", len(intra_pairs))
    intra_vectors = self._trainset_for(intra_pairs)
    logger.info("  -> Computing %d extrapersonal results", len(extra_pairs))
    extra_vectors = self._trainset_for(extra_pairs)

    logger.info("  -> Training BIC machine")
    trainer = bob.learn.linear.BICTrainer(self.M_I, self.M_E) if self.M_I is not None else bob.learn.linear.BICTrainer()
    trainer.train(intra_vectors, extra_vectors, self.bic_machine)

    # save the machine to file
    self.bic_machine.save(bob.io.base.HDF5File(enroller_file, 'w'))


  def load_enroller(self, enroller_file):
    """Reads the intrapersonal and extrapersonal mean and variance values"""
    self.bic_machine.load(bob.io.base.HDF5File(enroller_file, 'r'))
    # to set this should not be required, but just in case
    # you re-use a trained enroller file that hat different setup of use_DFFS
    self.bic_machine.use_DFFS = self.use_dffs


  def enroll(self, enroll_features):
    """Enrolls features by concatenating them"""
    return enroll_features


  def write_model(self, model, model_file):
    """Writes all features of the model into one HDF5 file, using the ``save_function`` specified in the constructor."""
    hdf5 = bob.io.base.HDF5File(model_file, "w")
    for i, f in enumerate(model):
      hdf5.create_group("Feature%d" % i)
      hdf5.cd("Feature%d" % i)
      self.write_function(f, hdf5)
      hdf5.cd("..")


  def read_model(self, model_file):
    """Loads all features of the model from the HDF5 file, using the ``load_function`` specified in the constructor."""
    hdf5 = bob.io.base.HDF5File(model_file)
    i = 0
    model = []
    while hdf5.has_group("Feature%d" % i):
      hdf5.cd("Feature%d" % i)
      model.append(self.read_function(hdf5))
      hdf5.cd("..")
      i += 1
    return model


  def read_probe(self, probe_file):
    """Loads the probe feature from file, using the ``load_function`` specified in the constructor."""
    return self.read_function(bob.io.base.HDF5File(probe_file))


  def score(self, model, probe):
    """Computes the IEC score for the given model and probe pair"""
    # compute average score for the models
    scores = []
    for i in range(len(model)):
      diff = self.comparison_function(model[i], probe)
      assert len(diff) == self.bic_machine.input_size
      scores.append(self.bic_machine(diff))
    return self.model_fusion_function(scores)
