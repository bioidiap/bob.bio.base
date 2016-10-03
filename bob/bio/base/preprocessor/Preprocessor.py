#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Tue Oct  2 12:12:39 CEST 2012

from .. import utils


class Preprocessor:
    """This is the base class for all preprocessors.
    It defines the minimum requirements for all derived proprocessor classes.

    **Parameters:**

    writes_data : bool
      Select, if the preprocessor actually writes preprocessed images, or if it is simply returning values.

    read_original_data: callable
      This function is used to read the original data from file.
      It takes three inputs: A :py:class:`bob.bio.base.database.BioFile` (or one of its derivatives), the original directory (as ``str``) and the original extension (as ``str``).

    kwargs : ``key=value`` pairs
      A list of keyword arguments to be written in the `__str__` function.
    """

    def __init__(self, writes_data=True, read_original_data=None, **kwargs):
        # Each class needs to have a constructor taking
        # all the parameters that are required for the preprocessing as arguments
        self.writes_data = writes_data
        if read_original_data is None:
            read_original_data = utils.read_original_data
        self.read_original_data = read_original_data
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

    def write_data(self, data, data_file):
        """Writes the given *preprocessed* data to a file with the given name.
        In this base class implementation, we simply use :py:func:`bob.bio.base.save` for that.
        If you have a different format (e.g. not images), please overwrite this function.

        **Parameters:**

        data : object
          The preprocessed data, i.e., what is returned from `__call__`.

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
