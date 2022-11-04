#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Set of legacy functionality for the bob.bio.base.database.Database interface.
"""

import logging
import os
import warnings

import bob.io.base

logger = logging.getLogger(__name__)


def convert_names_to_highlevel(names, low_level_names, high_level_names):
    """
    Converts group names from a low level to high level API

    This is useful for example when you want to return ``db.groups()`` for
    the :py:mod:`bob.bio.base`. Your instance of the database should
    already have ``low_level_names`` and ``high_level_names`` initialized.

    """

    if names is None:
        return None
    mapping = dict(zip(low_level_names, high_level_names))
    if isinstance(names, str):
        return mapping.get(names)
    return [mapping[g] for g in names]


def convert_names_to_lowlevel(names, low_level_names, high_level_names):
    """Same as :py:meth:`convert_names_to_highlevel` but on reverse"""

    if names is None:
        return None
    mapping = dict(zip(high_level_names, low_level_names))
    if isinstance(names, str):
        return mapping.get(names)
    return [mapping[g] for g in names]


def file_names(files, directory, extension):
    """file_names(files, directory, extension) -> paths

    Returns the full path of the given File objects.

    Parameters
    ----------
    files : list of ``bob.db.base.File``
        The list of file object to retrieve the file names for.

    directory : str
        The base directory, where the files can be found.

    extension : str
        The file name extension to add to all files.

    Returns
    -------
    paths : list of :obj:`str`
        The paths extracted for the files, in the same order.
    """
    # return the paths of the files, do not remove duplicates
    return [f.make_path(directory, extension) for f in files]


def sort_files(files):
    """Returns a sorted version of the given list of File's (or other structures
    that define an 'id' data member). The files will be sorted according to their
    id, and duplicate entries will be removed.

    Parameters
    ----------
    files : list of ``bob.db.base.File``
        The list of files to be uniquified and sorted.

    Returns
    -------
    sorted : list of ``bob.db.base.File``
        The sorted list of files, with duplicate `BioFile.id`\\s being removed.
    """
    # sort files using their sort function
    sorted_files = sorted(files)
    # remove duplicates
    return [
        f
        for i, f in enumerate(sorted_files)
        if not i or sorted_files[i - 1].id != f.id
    ]


def check_parameters_for_validity(
    parameters, parameter_description, valid_parameters, default_parameters=None
):
    """Checks the given parameters for validity.

    Checks a given parameter is in the set of valid parameters. It also
    assures that the parameters form a tuple or a list.  If parameters is
    'None' or empty, the default_parameters will be returned (if
    default_parameters is omitted, all valid_parameters are returned).

    This function will return a tuple or list of parameters, or raise a
    ValueError.


    Parameters
    ----------
    parameters : str or list of :obj:`str` or None
        The parameters to be checked. Might be a string, a list/tuple of
        strings, or None.

    parameter_description : str
        A short description of the parameter. This will be used to raise an
        exception in case the parameter is not valid.

    valid_parameters : list of :obj:`str`
        A list/tuple of valid values for the parameters.

    default_parameters : list of :obj:`str` or None
        The list/tuple of default parameters that will be returned in case
        parameters is None or empty. If omitted, all valid_parameters are used.

    Returns
    -------
    tuple
        A list or tuple containing the valid parameters.

    Raises
    ------
    ValueError
        If some of the parameters are not valid.

    """

    if not parameters:
        # parameters are not specified, i.e., 'None' or empty lists
        parameters = (
            default_parameters
            if default_parameters is not None
            else valid_parameters
        )

    if not isinstance(parameters, (list, tuple, set)):
        # parameter is just a single element, not a tuple or list -> transform it
        # into a tuple
        parameters = (parameters,)

    # perform the checks
    for parameter in parameters:
        if parameter not in valid_parameters:
            raise ValueError(
                "Invalid %s '%s'. Valid values are %s, or lists/tuples of those"
                % (parameter_description, parameter, valid_parameters)
            )

    # check passed, now return the list/tuple of parameters
    return parameters


def check_parameter_for_validity(
    parameter, parameter_description, valid_parameters, default_parameter=None
):
    """Checks the given parameter for validity

    Ensures a given parameter is in the set of valid parameters. If the
    parameter is ``None`` or empty, the value in ``default_parameter`` will
    be returned, in case it is specified, otherwise a :py:exc:`ValueError`
    will be raised.

    This function will return the parameter after the check tuple or list
    of parameters, or raise a :py:exc:`ValueError`.

    Parameters
    ----------
    parameter : :obj:`str` or :obj:`None`
                    The single parameter to be checked. Might be a string or None.

    parameter_description : str
                    A short description of the parameter. This will be used to raise an
                    exception in case the parameter is not valid.

    valid_parameters : list of :obj:`str`
                    A list/tuple of valid values for the parameters.

    default_parameter : list of :obj:`str`, optional
                    The default parameter that will be returned in case parameter is None or
                    empty. If omitted and parameter is empty, a ValueError is raised.

    Returns
    -------
    str
                    The validated parameter.

    Raises
    ------
    ValueError
                    If the specified parameter is invalid.

    """

    if parameter is None:
        # parameter not specified ...
        if default_parameter is not None:
            # ... -> use default parameter
            parameter = default_parameter
        else:
            # ... -> raise an exception
            raise ValueError(
                "The %s has to be one of %s, it might not be 'None'."
                % (parameter_description, valid_parameters)
            )

    if isinstance(parameter, (list, tuple, set)):
        # the parameter is in a list/tuple ...
        if len(parameter) > 1:
            raise ValueError(
                "The %s has to be one of %s, it might not be more than one "
                "(%s was given)."
                % (parameter_description, valid_parameters, parameter)
            )
        # ... -> we take the first one
        parameter = parameter[0]

    # perform the check
    if parameter not in valid_parameters:
        raise ValueError(
            "The given %s '%s' is not allowed. Please choose one of %s."
            % (parameter_description, parameter, valid_parameters)
        )

    # tests passed -> return the parameter
    return parameter


class File(object):
    """Abstract class that define basic properties of File objects.

    Your file instance should have at least the self.id and self.path
    properties.
    """

    def __init__(self, path, file_id=None, **kwargs):
        """**Constructor Documentation**

        Initialize the File object with the minimum required data.

        Parameters
        ----------
        path : str
            The path to this file, relative to the basic directory.
            If you use an SQL database, this should be the SQL type String.
            Please do not specify any file extensions.

        file_id : object
            The id of the file (various type). Its type depends on your
            implementation. If you use an SQL database, this should be an SQL type
            like Integer or String. If you are using an automatically determined
            file id, you don't need to specify this parameter.

        Raises
        ------
        NotImplementedError
            If self.id is not set and not specified during initialization through
            `file_id`.
        """

        self.path = path
        """A relative path, which includes file name but excludes file extension"""

        # set file id only, when specified
        if file_id:
            self.id = file_id
            """A unique identifier of the file."""
        else:
            # check that the file id at least exists
            if not hasattr(self, "id"):
                raise NotImplementedError(
                    "Please either specify the file id as parameter, or create an "
                    "'id' member variable in the derived class that is automatically "
                    "determined (e.g. by SQLite)"
                )

        super(File, self).__init__(**kwargs)

    def __lt__(self, other):
        """This function defines the order on the File objects. File objects are
        always ordered by their ID, in ascending order."""
        return self.id < other.id

    def __repr__(self):
        """This function describes how to convert a File object into a string."""
        return "<File('%s': '%s')>" % (str(self.id), str(self.path))

    def make_path(self, directory=None, extension=None):
        """Wraps the current path so that a complete path is formed

        Parameters
        ----------
        directory : :obj:`str`, optional
            An optional directory name that will be prefixed to the returned
            result.
        extension : :obj:`str`, optional
            An optional extension that will be suffixed to the returned filename.
            The extension normally includes the leading ``.`` character as in
            ``.jpg`` or ``.hdf5``.

        Returns
        -------
        str
            Returns a string containing the newly generated file path.
        """
        # assure that directory and extension are actually strings
        # create the path
        return str(os.path.join(directory or "", self.path + (extension or "")))

    def save(
        self, data, directory=None, extension=".hdf5", create_directories=True
    ):
        """Saves the input data at the specified location and using the given
        extension. Override it if you need to save differently.

        Parameters
        ----------
        data : object
            The data blob to be saved (normally a :py:class:`numpy.ndarray`).
        directory : :obj:`str`, optional
            If not empty or None, this directory is prefixed to the final
            file destination
        extension : :obj:`str`, optional
            The extension of the filename - this will control the type of
          output and the codec for saving the input blob.
        create_directories : :obj:`bool`, optional
            Whether to create the required directories to save the data.

        """
        # get the path
        path = self.make_path(directory or "", extension or "")
        # use the bob API to save the data
        bob.io.base.save(data, path, create_directories=create_directories)

    def load(self, directory=None, extension=".hdf5"):
        """Loads the data at the specified location and using the given extension.
        Override it if you need to load differently.

        Parameters
        ----------
        directory : :obj:`str`, optional
            If not empty or None, this directory is prefixed to the final
            file destination
        extension : :obj:`str`, optional
            If not empty or None, this extension is suffixed to the final
            file destination

        Returns
        -------
        object
            The loaded data (normally :py:class:`numpy.ndarray`).

        """
        # get the path
        path = self.make_path(directory or "", extension or "")
        return bob.io.base.load(path)


class FileDatabase(object):
    """Low-level File-based Database API to be used within Bob.

    Not all Databases in Bob need to inherit from this class. Use this class
    only if in your database one sample correlates to one actual file.

    Attributes
    ----------
    original_directory : str
        The directory where the raw files are located.
    original_extension : str
        The extension of raw data files, e.g. ``.png``.
    """

    def __init__(self, original_directory, original_extension, **kwargs):
        super(FileDatabase, self).__init__(**kwargs)
        self.original_directory = original_directory
        self.original_extension = original_extension

    def original_file_names(self, files):
        """Returns the full path of the original data of the given File objects.

        Parameters
        ----------
        files : list of ``bob.db.base.File``
            The list of file object to retrieve the original data file names for.

        Returns
        -------
        list of :obj:`str`
            The paths extracted for the files, in the same order.
        """
        if self.original_directory is None:
            logger.warning(
                "self.original_directory was not provided (must not be None)!"
            )
        if self.original_extension is None:
            logger.warning(
                "self.original_extension was not provided (must not be None)!"
            )
        return file_names(
            files, self.original_directory, self.original_extension
        )

    def original_file_name(self, file):
        """This function returns the original file name for the given File
        object.

        Parameters
        ----------
        file
            ``bob.db.base.File`` or a derivative
            The File objects for which the file name should be retrieved

        Returns
        -------
        str
            The original file name for the given ``bob.db.base.File``
            object.

        Raises
        ------
        ValueError
            if the file is not found.
        """
        # check if directory is set
        if not self.original_directory or not self.original_extension:
            logger.warning(
                "The original_directory and/or the original_extension were not"
                " specified in the constructor."
            )
        # extract file name
        file_name = file.make_path(
            self.original_directory, self.original_extension
        )

        if not self.check_existence or os.path.exists(file_name):
            return file_name

        raise ValueError(
            "The file '%s' was not found. Please check the "
            "original directory '%s' and extension '%s'?"
            % (file_name, self.original_directory, self.original_extension)
        )

    # Deprecated Methods below

    def check_parameters_for_validity(
        self,
        parameters,
        parameter_description,
        valid_parameters,
        default_parameters=None,
    ):
        warnings.warn(
            "check_parameters_for_validity is deprecated. Please use "
            "the equivalent function in this file",
            DeprecationWarning,
            stacklevel=2,
        )
        return check_parameters_for_validity(
            parameters,
            parameter_description,
            valid_parameters,
            default_parameters,
        )

    def check_parameter_for_validity(
        self,
        parameter,
        parameter_description,
        valid_parameters,
        default_parameter=None,
    ):
        warnings.warn(
            "check_parameter_for_validity is deprecated. Please use the "
            "equivalent function in this file",
            DeprecationWarning,
            stacklevel=2,
        )
        return check_parameter_for_validity(
            parameter,
            parameter_description,
            valid_parameters,
            default_parameter,
        )

    def convert_names_to_highlevel(
        self, names, low_level_names, high_level_names
    ):
        warnings.warn(
            "convert_names_to_highlevel is deprecated. Please use the "
            "equivalent function in this file",
            DeprecationWarning,
            stacklevel=2,
        )
        return convert_names_to_highlevel(
            names, low_level_names, high_level_names
        )

    def convert_names_to_lowlevel(
        self, names, low_level_names, high_level_names
    ):
        warnings.warn(
            "convert_names_to_lowlevel is deprecated. Please use the "
            "equivalent function in this file",
            DeprecationWarning,
            stacklevel=2,
        )
        return convert_names_to_lowlevel(
            names, low_level_names, high_level_names
        )

    def file_names(self, files, directory, extension):
        warnings.warn(
            "file_names is deprecated. Please use the "
            "equivalent function in this file",
            DeprecationWarning,
            stacklevel=2,
        )
        return file_names(files, directory, extension)

    def sort(self, files):
        warnings.warn(
            "sort is deprecated. Please use " "sort_files in bob.db.base.utils",
            DeprecationWarning,
            stacklevel=2,
        )
        return sort_files(files)


class Database(FileDatabase):
    """This class is deprecated. New databases should use the
    :py:class:`bob.db.base.FileDatabase` class if required"""

    def __init__(
        self, original_directory=None, original_extension=None, **kwargs
    ):
        warnings.warn(
            "The bob.db.base.Database class is deprecated. "
            "Please use bob.db.base.FileDatabase instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super(Database, self).__init__(
            original_directory, original_extension, **kwargs
        )
