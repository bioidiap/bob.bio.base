#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Tue Oct  2 12:12:39 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
import os
from .. import utils

class Algorithm:
  """This is the base class for all biometric recognition algorithms.
  It defines the minimum requirements for all derived algorithm classes.

  Call the constructor in derived class implementations.
  If your derived algorithm performs feature projection, please register this here.
  If it needs training for the projector or the enroller, please set this here, too.

  **Parameters:**

  performs_projection : bool
    Set to ``True`` if your derived algorithm performs a projection.
    Also implement the :py:meth:`project` function, and the :py:meth:`load_projector` if necessary.

  requires_projector_training : bool
    Only valid, when ``performs_projection = True``.
    Set this flag to ``False``, when the projection is applied, but the projector does not need to be trained.

  split_training_features_by_client : bool
    Only valid, when ``performs_projection = True`` and ``requires_projector_training = True``.
    If set to ``True``, the :py:meth:`train_projector` function will receive a double list (a list of lists) of data (sorted by identity).
    Otherwise, the :py:meth:`train_projector` function will receive data in a single list.

  use_projected_features_for_enrollment : bool
    Only valid, when ``performs_projection = True``.
    If set to false, the enrollment is performed using the original features, otherwise the features projected using the :py:meth:`project` function are used for model enrollment.

  requires_enroller_training : bool
    Set this flag to ``True``, when the enroller requires specialized training.
    Which kind of features are used for training depends on the ``use_projected_features_for_enrollment`` flag.

  multiple_model_scoring : str or ``None``
    The way, scores are fused when multiple features are stored in a one model.
    See :py:func:`bob.bio.base.score_fusion_strategy` for possible values.

  multiple_probe_scoring : str or ``None``
    The way, scores are fused when multiple probes are available.
    See :py:func:`bob.bio.base.score_fusion_strategy` for possible values.

  kwargs : ``key=value`` pairs
    A list of keyword arguments to be written in the :py:meth:`__str__` function.

  """

  def __init__(
      self,
      performs_projection = False, # enable if your tool will project the features
      requires_projector_training = True, # by default, the projector needs training, if projection is enabled
      split_training_features_by_client = False, # enable if your projector training needs the training files sorted by client
      use_projected_features_for_enrollment = True, # by default, the enroller used projected features for enrollment, if projection is enabled.
      requires_enroller_training = False, # enable if your enroller needs training

      multiple_model_scoring = 'average', # by default, compute the average between several models and the probe
      multiple_probe_scoring = 'average', # by default, compute the average between the model and several probes
      **kwargs                            # parameters from the derived class that should be reported in the __str__() function
  ):
    self.performs_projection = performs_projection
    self.requires_projector_training = performs_projection and requires_projector_training
    self.split_training_features_by_client = split_training_features_by_client
    self.use_projected_features_for_enrollment = performs_projection and use_projected_features_for_enrollment
    self.requires_enroller_training = requires_enroller_training
    self.model_fusion_function = utils.score_fusion_strategy(multiple_model_scoring)
    self.probe_fusion_function = utils.score_fusion_strategy(multiple_probe_scoring)
    self._kwargs = kwargs
    self._kwargs.update({'multiple_model_scoring':multiple_model_scoring, 'multiple_probe_scoring':multiple_probe_scoring})


  def __str__(self):
    """__str__() -> info

    This function returns all parameters of this class (and its derived class).

    **Returns:**

    info : str
      A string containing the full information of all parameters of this (and the derived) class.
    """
    return utils.pretty_print(self, self._kwargs)


  def project(self, feature):
    """project(feature) -> projected

    This function will project the given feature.
    It must be overwritten by derived classes, as soon as ``performs_projection = True`` was set in the constructor.
    It is assured that the :py:meth:`load_projector` was called once before the ``project`` function is executed.

    **Parameters:**

    feature : object
      The feature to be projected.

    **Returns:**

    projected : object
      The projected features.
      Must be writable with the :py:meth:`write_feature` function and readable with the :py:meth:`read_feature` function.

    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  def enroll(self, enroll_features):
    """enroll(enroll_features) -> model

    This function will enroll and return the model from the given list of features.
    It must be overwritten by derived classes.

    **Parameters:**

    enroll_features : [object]
      A list of features used for the enrollment of one model.

    **Returns:**

    model : object
      The model enrolled from the ``enroll_features``.
      Must be writable with the :py:meth:`write_model` function and readable with the :py:meth:`read_model` function.

    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  def score(self, model, probe):
    """score(model, probe) -> score

    This function will compute the score between the given model and probe.
    It must be overwritten by derived classes.

    **Parameters:**

    model : object
      The model to compare the probe with.
      The ``model`` was read using the :py:meth:`read_model` function.

    probe : object
      The probe object to compare the model with.
      The ``probe`` was read using the :py:meth:`read_probe` function.

    **Returns:**

    score : float
      A similarity between ``model`` and ``probe``.
      Higher values define higher similarities.
    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  def score_for_multiple_models(self, models, probe):
    """score_for_multiple_models(models, probe) -> score

    This function computes the score between the given model list and the given probe.
    In this base class implementation, it computes the scores for each model using the :py:meth:`score` method,
    and fuses the scores using the fusion method specified in the constructor of this class.
    Usually this function is called from derived class :py:meth:`score` functions.

    **Parameters:**

    models : [object]
      A list of model objects.

    probe : object
      The probe object to compare the models with.

    **Returns:**

    score : float
      The fused similarity between the given ``models`` and the ``probe``.
    """
    if isinstance(models, list):
      return self.model_fusion_function([self.score(model, probe) for model in models])
    elif isinstance(models, numpy.ndarray):
      return self.model_fusion_function([self.score(models[i,:], probe) for i in range(models.shape[0])])
    else:
      raise ValueError("The model does not have the desired format (list, array, ...)")


  def score_for_multiple_probes(self, model, probes):
    """score_for_multiple_probes(model, probes) -> score

    This function computes the score between the given model and the given probe files.
    In this base class implementation, it computes the scores for each probe file using the :py:meth:`score` method,
    and fuses the scores using the fusion method specified in the constructor of this class.

    **Parameters:**

    model : object
      A model object to compare the probes with.

    probes : [object]
      The list of probe object to compare the models with.

    **Returns:**

    score : float
      The fused similarity between the given ``model`` and the ``probes``.
    """
    if isinstance(probes, list):
      return self.probe_fusion_function([self.score(model, probe) for probe in probes])
    else:
      # only one probe feature -> use the default scoring function
      return self.score(model, probes)


  ############################################################
  ### Special functions that might be overwritten on need
  ############################################################

  def write_feature(self, feature, feature_file):
    """Saves the given *projected* feature to a file with the given name.
    In this base class implementation:

    - If the given feature has a ``save`` attribute, it calls ``feature.save(bob.io.base.HDF5File(feature_file), 'w')``.
      In this case, the given feature_file might be either a file name or a bob.io.base.HDF5File.
    - Otherwise, it uses :py:func:`bob.io.base.save` to do that.

    If you have a different format, please overwrite this function.

    Please register 'performs_projection = True' in the constructor to enable this function.

    **Parameters:**

    feature : object
      A feature as returned by the :py:meth:`project` function, which should be written.

    feature_file : str or :py:class:`bob.io.base.HDF5File`
      The file open for writing, or the file name to write to.
    """
    utils.save(feature, feature_file)


  def read_feature(self, feature_file):
    """read_feature(feature_file) -> feature

    Reads the *projected* feature from file.
    In this base class implementation, it uses :py:func:`bob.io.base.load` to do that.
    If you have different format, please overwrite this function.

    Please register ``performs_projection = True`` in the constructor to enable this function.

    **Parameters:**

    feature_file : str or :py:class:`bob.io.base.HDF5File`
      The file open for reading, or the file name to read from.

    **Returns:**

    feature : object
      The feature that was read from file.
    """
    return utils.load(feature_file)


  def write_model(self, model, model_file):
    """Writes the enrolled model to the given file.
    In this base class implementation:

    - If the given model has a 'save' attribute, it calls ``model.save(bob.io.base.HDF5File(model_file), 'w')``.
      In this case, the given model_file might be either a file name or a :py:class:`bob.io.base.HDF5File`.
    - Otherwise, it uses :py:func:`bob.io.base.save` to do that.

    If you have a different format, please overwrite this function.

    **Parameters:**

    model : object
      A model as returned by the :py:meth:`enroll` function, which should be written.

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The file open for writing, or the file name to write to.
    """
    utils.save(model, model_file)


  def read_model(self, model_file):
    """read_model(model_file) -> model

    Loads the enrolled model from file.
    In this base class implementation, it uses :py:func:`bob.io.base.load` to do that.

    If you have a different format, please overwrite this function.

    **Parameters:**

    model_file : str or :py:class:`bob.io.base.HDF5File`
      The file open for reading, or the file name to read from.

    **Returns:**

    model : object
      The model that was read from file.
    """
    return utils.load(model_file)


  def read_probe(self, probe_file):
    """read_probe(probe_file) -> probe

    Reads the probe feature from file.
    By default, the probe feature is identical to the projected feature.
    Hence, this base class implementation simply calls :py:meth:`read_feature`.

    If your algorithm requires different behavior, please overwrite this function.

    **Parameters:**

    probe_file : str or :py:class:`bob.io.base.HDF5File`
      The file open for reading, or the file name to read from.

    **Returns:**

    probe : object
      The probe that was read from file.
    """
    return self.read_feature(probe_file)



  def train_projector(self, training_features, projector_file):
    """This function can be overwritten to train the feature projector.
    If you do this, please also register the function by calling this base class constructor
    and enabling the training by ``requires_projector_training = True``.

    **Parameters:**

    training_features : [object] or [[object]]
      A list of *extracted* features that can be used for training the projector.
      Features will be provided in a single list, if ``split_training_features_by_client = False`` was specified in the constructor,
      otherwise the features will be split into lists, each of which contains the features of a single (training-)client.

    projector_file : str
      The file to write.
      This file should be readable with the :py:meth:`load_projector` function.
    """
    raise NotImplementedError("Please overwrite this function in your derived class, or unset the 'requires_projector_training' option in the constructor.")


  def load_projector(self, projector_file):
    """Loads the parameters required for feature projection from file.
    This function usually is useful in combination with the :py:meth:`train_projector` function.
    In this base class implementation, it does nothing.

    Please register `performs_projection = True` in the constructor to enable this function.

    **Parameters:**

    projector_file : str
      The file to read the projector from.
    """
    pass


  def train_enroller(self, training_features, enroller_file):
    """This function can be overwritten to train the model enroller.
    If you do this, please also register the function by calling this base class constructor
    and enabling the training by ``require_enroller_training = True``.

    **Parameters:**

    training_features : [object] or [[object]]
      A list of *extracted* features that can be used for training the projector.
      Features will be split into lists, each of which contains the features of a single (training-)client.

    enroller_file : str
      The file to write.
      This file should be readable with the :py:meth:`load_enroller` function.
    """


  def load_enroller(self, enroller_file):
    """Loads the parameters required for model enrollment from file.
    This function usually is only useful in combination with the :py:meth:`train_enroller` function.
    This function is always called **after** calling :py:meth:`load_projector`.
    In this base class implementation, it does nothing.

    **Parameters:**

    enroller_file : str
      The file to read the enroller from.
    """
    pass
