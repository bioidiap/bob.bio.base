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
  """Computes the Intrapersonal/Extrapersonal classifier using a generic feature type and feature comparison function.

  In this generic implementation, any distance or similarity vector that results as a comparison of two feature vectors can be used.
  Currently two different versions are implemented: One with [MWP98]_ and one without (a generalization of [GW09]_) subspace projection of the features.
  The implementation of the BIC training is taken from :ref:`bob.learn.linear <bob.learn.linear>`.

  **Parameters:**

  comparison_function : function
    The function to compare the features in the original feature space.
    For a given pair of features, this function is supposed to compute a vector of similarity (or distance) values.
    In the easiest case, it just computes the element-wise difference of the feature vectors, but more difficult functions can be applied, and the function might be specialized for the features you put in.

  maximum_training_pair_count : int or None
    Limit the number of training image pairs to the given value, i.e., to avoid memory issues.

  subspace_dimensions : (int, int) or None
    A tuple of sizes of the intrapersonal and extrapersonal subspaces.
    If given, subspace projection is performed (cf. [MWP98]_) and the subspace projection matrices are truncated to the given sizes.
    If omitted, no subspace projection is performed (cf. [GW09]_).

  uses_dffs : bool
    Only valid, when ``subspace_dimensions`` are specified.
    Use the *Distance From Feature Space* (DFFS) (cf. [MWP98]_) during scoring.
    Use this flag with care!

  read_function : function
    A function to read a feature from :py:class:`bob.io.base.HDF5File`.
    This function need to be appropriate to read the type of features that you are using.
    By default, :py:func:`bob.bio.base.load` is used.

  write_function : function
    A function to write a feature to :py:class:`bob.io.base.HDF5File`.
    This function is used to write the model and need to be appropriate to write the type of features that you are using.
    By default, :py:func:`bob.bio.base.save` is used.

  kwargs : ``key=value`` pairs
    A list of keyword arguments directly passed to the :py:class:`Algorithm` base class constructor.
  """

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


  def _trainset_for(self, pairs):
    """Computes the array containing the comparison results for the given set of image pairs."""
    return numpy.vstack([self.comparison_function(f1, f2) for (f1, f2) in pairs])


  def train_enroller(self, train_features, enroller_file):
    """Trains the BIC by computing intra-personal and extra-personal subspaces.

    First, two lists of pairs are computed, which contain intra-personal and extra-personal feature pairs, respectively.
    Afterward, the comparison vectors are computed using the ``comparison_function`` specified in the constructor.
    Finally, the :py:class:`bob.learn.linear.BICTrainer` is used to train a :py:class:`bob.learn.linear.BICMachine`.

    **Parameters:**

    train_features : [[object]]
      A list of lists of feature vectors, which are used to train the BIC.
      Each sub-list contains the features of one client.

    enroller_file : str
      A writable file, into which the resulting :py:class:`bob.learn.linear.BICMachine` will be written.
    """

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
    """Reads the :py:class:`bob.learn.linear.BICMachine` from file.

    The :py:attr:`bob.learn.linear.BICMachine.use_DFFS` will be overwritten by the ``use_dffs`` value specified in this class' constructor.

    **Parameters:**

    enroller_file : str
      An existing file, from which the :py:class:`bob.learn.linear.BICMachine` will be read.
    """
    self.bic_machine.load(bob.io.base.HDF5File(enroller_file, 'r'))
    # to set this should not be required, but just in case
    # you re-use a trained enroller file that hat different setup of use_DFFS
    self.bic_machine.use_DFFS = self.use_dffs


  def enroll(self, enroll_features):
    """enroll(enroll_features) -> model

    Enrolls the model by storing all given input features.
    The features must be writable with the ``write_function`` defined in the constructor.

    **Parameters:**

    enroll_features : [object]
      The list of projected features to enroll the model from.

    **Returns:**

    model : [object]
      The enrolled model (which is identical to the input features).
    """
    return enroll_features


  def write_model(self, model, model_file):
    """Writes all features of the model into one HDF5 file.

    To write the features, the ``write_function`` specified in the constructor is employed.

    **Parameters:**

    model : [object]
      The model to write, which is a list of features.

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The file (open for writing) or a file name to write into.
    """
    hdf5 = model_file if isinstance(model_file, bob.io.base.HDF5File) else bob.io.base.HDF5File(model_file, 'w')
    for i, f in enumerate(model):
      hdf5.create_group("Feature%d" % i)
      hdf5.cd("Feature%d" % i)
      self.write_function(f, hdf5)
      hdf5.cd("..")


  def read_model(self, model_file):
    """read_model(model_file) -> model

    Reads all features of the model from the given HDF5 file.

    To read the features, the ``read_function`` specified in the constructor is employed.

    **Parameters:**

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The file (open for reading) or the name of an existing file to read from.

    **Returns:**

    model : [object]
      The read model, which is a list of features.
    """
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
    """read_probe(probe_file) -> probe

    Reads the probe feature from the given HDF5 file.

    To read the feature, the ``read_function`` specified in the constructor is employed.

    **Parameters:**

    probe_file : str or :py:class:`bob.io.base.HDF5File`
      The file (open for reading) or the name of an existing file to read from.

    **Returns:**

    probe : object
      The read probe, which is a feature.
    """
    return self.read_function(bob.io.base.HDF5File(probe_file))


  def score(self, model, probe):
    """score(model, probe) -> float

    Computes the BIC score between the model and the probe.
    First, the ``comparison_function`` is used to create the comparison vectors between all model features and the probe feature.
    Then, a BIC score is computed for each comparison vector, and the BIC scores are fused using the :py:func:`model_fusion_function` defined in the :py:class:`Algorithm` base class.

    **Parameters:**

    model : [object]
      The model storing all model features.

    probe : object
      The probe feature.

    **Returns:**

    score : float
      A fused BIC similarity value between ``model`` and ``probe``.
    """
    # compute average score for the models
    scores = []
    for i in range(len(model)):
      diff = self.comparison_function(model[i], probe)
      assert len(diff) == self.bic_machine.input_size
      scores.append(self.bic_machine(diff))
    return self.model_fusion_function(scores)

  # re-define unused functions, just so that they do not get documented
  def train_projector(*args,**kwargs): raise NotImplementedError()
  def load_projector(*args,**kwargs): pass
  def project(*args,**kwargs): raise NotImplementedError()
  def write_feature(*args,**kwargs): raise NotImplementedError()
  def read_feature(*args,**kwargs): raise NotImplementedError()
