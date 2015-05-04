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

import os

from .. import utils

class Extractor:
  """This is the base class for all feature extractors.
  It defines the minimum requirements that a derived feature extractor class need to implement.

  The constructor takes two parameters:

  requires_training : bool
    Set this flag to ``True`` if your feature extractor needs to be trained.
    In that case, please override the :py:meth:`train` and :py:meth:`load` methods

  split_training_data_by_client : bool
    Set this flag to ``True`` if your feature extractor requires the training data to be split by clients.
    Ignored, if ``requires_training`` is ``False``
  """

  def __init__(
      self,
      requires_training = False, # enable, if your extractor needs training
      split_training_data_by_client = False, # enable, if your extractor needs the training files sorted by client
      **kwargs                   # the parameters of the extractor, to be written in the __str__() method
  ):
    # Each class needs to have a constructor taking
    # all the parameters that are required for the feature extraction as arguments
    self.requires_training = requires_training
    self.split_training_data_by_client = split_training_data_by_client
    self._kwargs = kwargs


  ############################################################
  ### functions that must be overwritten in derived classes
  ############################################################

  def __call__(self, data):
    """This function will actually perform the feature extraction.
    It must be overwritten by derived classes.
    It takes the (preprocessed) data and returns the features extracted from the data.
    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  def __str__(self):
    """This function returns a string containing all parameters of this class (and its derived class)."""
    return "%s(%s)" % (str(self.__class__), ", ".join(["%s=%s" % (key, value) for key,value in self._kwargs.items() if value is not None]))


  ############################################################
  ### Special functions that might be overwritten on need
  ############################################################

  def write_feature(self, feature, feature_file):
    """Writes the given *extracted* feature to a file with the given name.
    In this base class implementation, we simply use :py:func:`bob.bio.base.save` for that.
    If you have a different format, please overwrite this function.
    """
    utils.save(feature, feature_file)


  def read_feature(self, feature_file):
    """Reads the *extracted* feature from file.
    In this base class implementation, it uses :py:func:`bob.bio.base.load` to do that.
    If you have different format, please overwrite this function.
    """
    return utils.load(feature_file)


  def load(self, extractor_file):
    """Loads the parameters required for feature extraction from the extractor file.
    This function usually is only useful in combination with the 'train' function (see below).
    In this base class implementation, it does nothing.
    """
    pass


  def train(self, data_list, extractor_file):
    """This function can be overwritten to train the feature extractor.
    If you do this, please also register the function by calling this base class constructor
    and enabling the training by 'requires_training = True'.

    The training function gets two parameters:

    - data_list: A list of data that can be used for training the extractor.
    - extractor_file: The file to write. This file should be readable with the 'load' function (see above).
    """
    raise NotImplementedError("Please overwrite this function in your derived class, or unset the 'requires_training' option in the constructor.")
