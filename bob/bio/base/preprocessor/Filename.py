# @date Wed May 11 12:39:37 MDT 2016
# @author Manuel Gunther <siebenkopf@googlemail.com>

import bob.io.base

import os

from .Preprocessor import Preprocessor

class Filename (Preprocessor):
  """This preprocessor is simply passing over the file name, in order to be used in an extractor that loads the data from file.

  The file name that will be returned by the :py:meth:`read_data` function will contain the path of the :py:class:`bob.bio.db.BioFile`, but it might contain more paths (such as the ``--preprocessed-directory`` passed on command line).
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
