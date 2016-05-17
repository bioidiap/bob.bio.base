# @date Wed May 11 12:39:37 MDT 2016
# @author Manuel Gunther <siebenkopf@googlemail.com>
#
# Copyright (c) 2016, Regents of the University of Colorado on behalf of the University of Colorado Colorado Springs.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import bob.io.base

import os

from .Preprocessor import Preprocessor

class Filename (Preprocessor):
  """This preprocessor is simply passing over the file name, in order to be used in an extractor that loads the data from file.

  The file name that will be returned by the :py:meth:`read_data` function will contain the path of the :py:class:`bob.db.verification.utils.File`, but it might contain more paths (such as the ``--preprocessed-directory`` passed on command line).
  """

  def __init__(self):
    Preprocessor.__init__(self, writes_data=False)


  # The call function (i.e. the operator() in C++ terms)
  def __call__(self, data, annotations = None):
    """__call__(data, annotations) -> data

    This function appears to do something, but it simply returns ``1``, which is used nowhere.
    We could also return ``None``, but this might trigger warnings in the calling function.

    **Parameters:**

    ``data`` : ``None``
      The file name returned by :py:meth:`read_original_data`.

    ``annotations`` : any
      ignored.

    **Returns:**

    ``data`` : int
      1 throughout
    """
    return 1


  ############################################################
  ### Special functions that might be overwritten on need
  ############################################################

  def read_original_data(self, original_file_name):
    """read_original_data(original_file_name) -> data

    This function does **not** read the original image..

    **Parameters:**

    ``original_file_name`` : any
      ignored

    **Returns:**

    ``data`` : ``None``
      throughout.
    """
    pass


  def write_data(self, data, data_file):
    """Does **not** write any data.

    ``data`` : any
      ignored.

    ``data_file`` : any
      ignored.
    """
    pass


  def read_data(self, data_file):
    """read_data(data_file) -> data

    Returns the name of the data file without its filename extension.

    **Parameters:**

    ``data_file`` : str
      The name of the preprocessed data file.

    **Returns:**

    ``data`` : str
      The preprocessed data read from file.
    """
    return os.path.splitext(data_file)[0]
