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

import bob.io.base

import os

from .. import utils

class Preprocessor:
  """This is the base class for all preprocessors.
  It defines the minimum requirements for all derived proprocessor classes.

  **Parameters:**

  writes_data : bool
    Select, if the preprocessor actually writes preprocessed images, or if it is simply returning values.

  kwargs : ``key=value`` pairs
    A list of keyword arguments to be written in the :py:meth:`__str__` function.
  """

  def __init__(self, writes_data = True, **kwargs):
    # Each class needs to have a constructor taking
    # all the parameters that are required for the preprocessing as arguments
    self.writes_data = writes_data
    self._kwargs = kwargs
    pass


  # The call function (i.e. the operator() in C++ terms)
  def __call__(self, data, annotations):
    """__call__(data, annotations) -> dara

    This is the call function that you have to overwrite in the derived class.
    The parameters that this function will receive are:

    **Parameters:**

    data : object
      The original data that needs preprocessing, usually a :py:class:`numpy.ndarray`, but might be different.

    annotations : {} or None
      The annotations (if any)  that belongs to the given ``data``; as a dictionary.
      The type of the annotation depends on your kind of problem.

    **Returns:**

    data : object
      The *preprocessed* data, usually a :py:class:`numpy.ndarray`, but might be different.
    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  def __str__(self):
    """__str__() -> info

    This function returns all parameters of this class (and its derived class).

    **Returns:**

    info : str
      A string containing the full information of all parameters of this (and the derived) class.
    """
    return utils.pretty_print(self, self._kwargs)

  ############################################################
  ### Special functions that might be overwritten on need
  ############################################################

  def read_original_data(self, original_file_name):
    """read_original_data(original_file_name) -> data

    Reads the *original* data (usually something like an image) from file.
    In this base class implementation, it uses :py:func:`bob.io.base.load` to do that.
    If you have different format, please overwrite this function.

    **Parameters:**

    original_file_name : str
      The file name to read the original data from.

    **Returns:**

    data : object (usually :py:class:`numpy.ndarray`)
      The original data read from file.
    """
    return bob.io.base.load(original_file_name)


  def write_data(self, data, data_file):
    """Writes the given *preprocessed* data to a file with the given name.
    In this base class implementation, we simply use :py:func:`bob.bio.base.save` for that.
    If you have a different format (e.g. not images), please overwrite this function.

    **Parameters:**

    data : object
      The preprocessed data, i.e., what is returned from :py:meth:`__call__`.

    data_file : str or :py:class:`bob.io.base.HDF5File`
      The file open for writing, or the name of the file to write.
    """
    utils.save(data, data_file)


  def read_data(self, data_file):
    """read_data(data_file) -> data

    Reads the *preprocessed* data from file.
    In this base class implementation, it uses :py:func:`bob.bio.base.load` to do that.
    If you have different format, please overwrite this function.

    **Parameters:**

    data_file : str or :py:class:`bob.io.base.HDF5File`
      The file open for reading or the name of the file to read from.

    **Returns:**

    data : object (usually :py:class:`numpy.ndarray`)
      The preprocessed data read from file.
    """
    return utils.load(data_file)
