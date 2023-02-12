#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Tue Oct  2 12:12:39 CEST 2012

import warnings

from .. import utils


class Extractor(object):
    """This is the base class for all feature extractors.
    It defines the minimum requirements that a derived feature extractor class need to implement.

    If your derived class requires training, please register this here.

    **Parameters**

    requires_training : bool
      Set this flag to ``True`` if your feature extractor needs to be trained.
      In that case, please override the :py:meth:`train` and :py:meth:`load` methods

    split_training_data_by_client : bool
      Set this flag to ``True`` if your feature extractor requires the training data to be split by clients.
      Ignored, if ``requires_training`` is ``False``

    min_extractor_file_size : int
        The minimum file size of a saved extractor file for extractors that
        require training in bytes. If the saved file size is smaller than this, it
        is assumed to be a corrupt file and the extractor will be trained again.

    min_feature_file_size : int
        The minimum file size of extracted features in bytes. If the saved file
        size is smaller than this, it is assumed to be a corrupt file and the
        features will be extracted again.

    kwargs : ``key=value`` pairs
      A list of keyword arguments to be written in the `__str__` function.
    """

    def __init__(
        self,
        requires_training=False,  # enable, if your extractor needs training
        split_training_data_by_client=False,  # enable, if your extractor needs the training files sorted by client
        min_extractor_file_size=1000,
        min_feature_file_size=1000,
        **kwargs,  # the parameters of the extractor, to be written in the __str__() method
    ):
        # Each class needs to have a constructor taking
        # all the parameters that are required for the feature extraction as arguments
        self.requires_training = requires_training
        self.split_training_data_by_client = split_training_data_by_client
        self.min_extractor_file_size = min_extractor_file_size
        self.min_feature_file_size = min_feature_file_size
        self._kwargs = kwargs

        warnings.warn(
            "`bob.bio.base.extractor.Extractor` will be deprecated in 01/01/2021. "
            "Please, implement your biometric algorithm using `bob.pipelines` (https://gitlab.idiap.ch/bob/bob.pipelines).",
            DeprecationWarning,
        )

    ############################################################
    # functions that must be overwritten in derived classes
    ############################################################

    def __call__(self, data):
        """__call__(data) -> feature

        This function will actually perform the feature extraction.
        It must be overwritten by derived classes.
        It takes the (preprocessed) data and returns the features extracted from the data.

        **Parameters**

        data : object (usually :py:class:`numpy.ndarray`)
          The *preprocessed* data from which features should be extracted.

        **Returns:**

        feature : object (usually :py:class:`numpy.ndarray`)
          The extracted feature.
        """
        raise NotImplementedError(
            "Please overwrite this function in your derived class"
        )

    def __str__(self):
        """__str__() -> info

        This function returns all parameters of this class (and its derived class).

        **Returns:**

        info : str
          A string containing the full information of all parameters of this (and the derived) class.
        """
        return utils.pretty_print(self, self._kwargs)

    ############################################################
    # Special functions that might be overwritten on need
    ############################################################

    def write_feature(self, feature, feature_file):
        """Writes the given *extracted* feature to a file with the given name.
        In this base class implementation, we simply use :py:func:`bob.bio.base.save` for that.
        If you have a different format, please overwrite this function.

        **Parameters:**

        feature : object
          The extracted feature, i.e., what is returned from `__call__`.

        feature_file : str or :py:class:`h5py.File`
          The file open for writing, or the name of the file to write.
        """
        utils.save(feature, feature_file)

    def read_feature(self, feature_file):
        """Reads the *extracted* feature from file.
        In this base class implementation, it uses :py:func:`bob.bio.base.load` to do that.
        If you have different format, please overwrite this function.

        **Parameters:**

        feature_file : str or :py:class:`h5py.File`
          The file open for reading or the name of the file to read from.

        **Returns:**

        feature : object (usually :py:class:`numpy.ndarray`)
          The feature read from file.
        """
        return utils.load(feature_file)

    def load(self, extractor_file):
        """Loads the parameters required for feature extraction from the extractor file.
        This function usually is only useful in combination with the :py:meth:`train` function.
        In this base class implementation, it does nothing.

        **Parameters:**

        extractor_file : str
          The file to read the extractor from.
        """
        pass

    def train(self, training_data, extractor_file):
        """This function can be overwritten to train the feature extractor.
        If you do this, please also register the function by calling this base class constructor
        and enabling the training by ``requires_training = True``.

        **Parameters:**

        training_data : [object] or [[object]]
          A list of *preprocessed* data that can be used for training the extractor.
          Data will be provided in a single list, if ``split_training_features_by_client = False`` was specified in the constructor,
          otherwise the data will be split into lists, each of which contains the data of a single (training-)client.

        extractor_file : str
          The file to write.
          This file should be readable with the :py:meth:`load` function.
        """
        raise NotImplementedError(
            "Please overwrite this function in your derived class, or unset the 'requires_training' option in the constructor."
        )
