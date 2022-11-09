#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import abc
import os

from .file import BioFile
from .legacy import FileDatabase as LegacyFileDatabase


class BioDatabase(LegacyFileDatabase, metaclass=abc.ABCMeta):
    """This class represents the basic API for database access.
    Please use this class as a base class for your database access classes.
    Do not forget to call the constructor of this base class in your derived class.

    **Parameters:**

    name : str
    A unique name for the database.

    all_files_options : dict
    Dictionary of options passed to the :py:meth:`bob.bio.base.database.BioDatabase.objects` database query when retrieving all data.

    extractor_training_options : dict
    Dictionary of options passed to the :py:meth:`bob.bio.base.database.BioDatabase.objects` database query used to retrieve the files for the extractor training.

    projector_training_options : dict
    Dictionary of options passed to the :py:meth:`bob.bio.base.database.BioDatabase.objects` database query used to retrieve the files for the projector training.

    enroller_training_options : dict
    Dictionary of options passed to the :py:meth:`bob.bio.base.database.BioDatabase.objects` database query used to retrieve the files for the enroller training.

    check_original_files_for_existence : bool
    Enables to test for the original data files when querying the database.

    original_directory : str
    The directory where the original data of the database are stored.

    original_extension : str
    The file name extension of the original data.

    annotation_directory : str
    The directory where the image annotations of the database are stored, if any.

    annotation_extension : str
    The file name extension of the annotation files.

    annotation_type : str
    The type of the annotation file to read, only json works.

    protocol : str or ``None``
    The name of the protocol that defines the default experimental setup for this database.

    training_depends_on_protocol : bool
    Specifies, if the training set used for training the extractor and the projector depend on the protocol.
    This flag is used to avoid re-computation of data when running on the different protocols of the same database.

    models_depend_on_protocol : bool
    Specifies, if the models depend on the protocol.
    This flag is used to avoid re-computation of models when running on the different protocols of the same database.

    kwargs : ``key=value`` pairs
    The arguments of the `Database` base class constructor.

    """

    # tell test runners (such as nose and pytest) that this class is not a test class
    ___test___ = False

    def __init__(
        self,
        name,
        all_files_options={},  # additional options for the database query that can be used to extract all files
        extractor_training_options={},
        # additional options for the database query that can be used to extract the training files for the extractor training
        projector_training_options={},
        # additional options for the database query that can be used to extract the training files for the extractor training
        enroller_training_options={},
        # additional options for the database query that can be used to extract the training files for the extractor training
        check_original_files_for_existence=False,
        original_directory=None,
        original_extension=None,
        annotation_directory=None,
        annotation_extension=None,
        annotation_type=None,
        protocol="Default",
        training_depends_on_protocol=False,
        models_depend_on_protocol=False,
        **kwargs
    ):

        assert isinstance(name, str)

        super(BioDatabase, self).__init__(
            original_directory=original_directory,
            original_extension=original_extension,
            **kwargs
        )

        self.name = name

        self.all_files_options = all_files_options
        self.extractor_training_options = extractor_training_options
        self.projector_training_options = projector_training_options
        self.enroller_training_options = enroller_training_options
        self.check_existence = check_original_files_for_existence

        self._kwargs = {}

        self.annotation_directory = annotation_directory
        self.annotation_extension = annotation_extension or ".json"
        self.annotation_type = annotation_type or "json"
        self.protocol = protocol
        self.training_depends_on_protocol = training_depends_on_protocol
        self.models_depend_on_protocol = models_depend_on_protocol
        self.models_depend_on_protocol = models_depend_on_protocol

        # try if the implemented model_ids_with_protocol() and objects() function have at least the required interface
        try:
            # create a value that is very unlikely a valid value for anything
            test_value = "#6T7+§X"
            # test if the parameters of the functions apply
            self.model_ids_with_protocol(groups=test_value, protocol=test_value)
            self.objects(
                groups=test_value,
                protocol=test_value,
                purposes=test_value,
                model_ids=(test_value,),
            )
            self.annotations(file=BioFile(test_value, test_value, test_value))
        except TypeError as e:
            # type error indicates that the given parameters are not valid.
            raise NotImplementedError(
                str(e)
                + "\nPlease implement:\n - the model_ids_with_protocol(...) function with at least the "
                "arguments 'groups' and 'protocol'\n - the objects(...) function with at least the "
                "arguments 'groups', 'protocol', 'purposes' and 'model_ids'\n - the annotations() "
                "function with at least the arguments 'file_id'."
            )
        except Exception:
            # any other error is fine at this stage.
            pass

    def __str__(self):
        """__str__() -> info

        This function returns all parameters of this class.

        **Returns:**

        info : str
          A string containing the full information of all parameters of this class.
        """
        params = (
            "name=%s, protocol=%s, original_directory=%s, original_extension=%s"
            % (
                self.name,
                self.protocol,
                self.original_directory,
                self.original_extension,
            )
        )
        params += ", ".join(
            ["%s=%s" % (key, value) for key, value in self._kwargs.items()]
        )
        params += ", original_directory=%s, original_extension=%s" % (
            self.original_directory,
            self.original_extension,
        )
        if self.all_files_options:
            params += ", all_files_options=%s" % self.all_files_options
        if self.extractor_training_options:
            params += (
                ", extractor_training_options=%s"
                % self.extractor_training_options
            )
        if self.projector_training_options:
            params += (
                ", projector_training_options=%s"
                % self.projector_training_options
            )
        if self.enroller_training_options:
            params += (
                ", enroller_training_options=%s"
                % self.enroller_training_options
            )

        return "%s(%s)" % (str(self.__class__), params)

    def replace_directories(self, replacements=None):
        """This helper function replaces the ``original_directory`` and the ``annotation_directory`` of the database with the directories read from the given replacement file.

        This function is provided for convenience, so that the database configuration files do not need to be modified.
        Instead, this function uses the given dictionary of replacements to change the original directory and the original extension (if given).

        The given ``replacements`` can be of type ``dict``, including all replacements, or a file name (as a ``str``), in which case the file is read.
        The structure of the file should be:

        .. code-block:: text

           # Comments starting with # and empty lines are ignored

           [YOUR_..._DATA_DIRECTORY] = /path/to/your/data
           [YOUR_..._ANNOTATION_DIRECTORY] = /path/to/your/annotations

        If no annotation files are available (e.g. when they are stored inside the ``database``), the annotation directory can be left out.

        **Parameters:**

        replacements : dict or str
          A dictionary with replacements, or a name of a file to read the dictionary from.
          If the file name does not exist, no directories are replaced.
        """
        if replacements is None:
            return
        if isinstance(replacements, str):
            if not os.path.exists(replacements):
                return
            # Open the database replacement file and reads its content
            with open(replacements) as f:
                replacements = {}
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        splits = line.split("=")
                        assert len(splits) == 2
                        replacements[splits[0].strip()] = splits[1].strip()

        assert isinstance(replacements, dict)

        if self.original_directory in replacements:
            self.original_directory = replacements[self.original_directory]

        try:
            if self.annotation_directory in replacements:
                self.annotation_directory = replacements[
                    self.annotation_directory
                ]
        except AttributeError:
            pass

    ###########################################################################
    # Helper functions that you might want to use in derived classes
    ###########################################################################
    def uses_probe_file_sets(self, protocol=None):
        """Defines if, for the current protocol, the database uses several probe files to generate a score.
        Returns True if the given protocol specifies file sets for probes, instead of a single probe file.
        In this default implementation, False is returned, throughout.
        If you need different behavior, please overload this function in your derived class."""
        return False

    def arrange_by_client(self, files):
        """arrange_by_client(files) -> files_by_client

        Arranges the given list of files by client id.
        This function returns a list of lists of File's.

        **Parameters:**

        files : :py:class:`bob.bio.base.database.BioFile`
          A list of files that should be split up by `BioFile.client_id`.

        **Returns:**

        files_by_client : [[:py:class:`bob.bio.base.database.BioFile`]]
          The list of lists of files, where each sub-list groups the files with the same `BioFile.client_id`
        """
        client_files = {}
        for file in files:
            if file.client_id not in client_files:
                client_files[file.client_id] = []
            client_files[file.client_id].append(file)

        files_by_clients = []
        for client in sorted(client_files.keys()):
            files_by_clients.append(client_files[client])
        return files_by_clients

    def file_names(self, files, directory, extension):
        """file_names(files, directory, extension) -> paths

        Returns the full path of the given File objects.

        **Parameters:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The list of file object to retrieve the file names for.

        directory : str
          The base directory, where the files can be found.

        extension : str
          The file name extension to add to all files.

        **Returns:**

        paths : [str] or [[str]]
          The paths extracted for the files, in the same order.
          If this database provides file sets, a list of lists of file names is returned, one sub-list for each file set.
        """
        # return the paths of the files
        if self.uses_probe_file_sets() and files and hasattr(files[0], "files"):
            # List of Filesets: do not remove duplicates
            return [
                [f.make_path(directory, extension) for f in file_set.files]
                for file_set in files
            ]
        else:
            # List of files, do not remove duplicates
            return [f.make_path(directory, extension) for f in files]

    #################################################################
    # Methods to be overwritten by derived classes
    #################################################################
    @abc.abstractmethod
    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        """model_ids_with_protocol(groups = None, protocol = None, **kwargs) -> ids

        Returns a list of model ids for the given groups and given protocol.

        **Parameters:**

        groups : one or more of ``('world', 'dev', 'eval')``
          The groups to get the model ids for.

        protocol: a protocol name

        **Returns:**

        ids : [int] or [str]
          The list of (unique) model ids for the given groups.
        """
        raise NotImplementedError(
            "Please implement this function in derived classes"
        )

    def groups(self, protocol=None):
        """
        Returns the names of all registered groups in the database

        Keyword parameters:

        protocol: str
          The protocol for which the groups should be retrieved.
          If you do not have protocols defined, just ignore this field.
        """
        raise NotImplementedError(
            "This function must be implemented in your derived class."
        )

    @abc.abstractmethod
    def objects(
        self,
        groups=None,
        protocol=None,
        purposes=None,
        model_ids=None,
        **kwargs
    ):
        """This function returns a list of :py:class:`bob.bio.base.database.BioFile` objects or the list
        of objects which inherit from this class. Returned files fulfill the given restrictions.

        Keyword parameters:

        groups : str or [str]
          The groups of which the clients should be returned.
          Usually, groups are one or more elements of ('world', 'dev', 'eval')

        protocol
          The protocol for which the clients should be retrieved.
          The protocol is dependent on your database.
          If you do not have protocols defined, just ignore this field.

        purposes : str or [str]
          The purposes for which File objects should be retrieved.
          Usually, purposes are one of ('enroll', 'probe').

        model_ids : [various type]
          The model ids for which the File objects should be retrieved.
          What defines a 'model id' is dependent on the database.
          In cases, where there is only one model per client, model ids and client ids are identical.
          In cases, where there is one model per file, model ids and file ids are identical.
          But, there might also be other cases.
        """
        raise NotImplementedError(
            "This function must be implemented in your derived class."
        )

    def annotations(self, file):
        """
        Returns the annotations for the given File object, if available.
        You need to override this method in your high-level implementation.
        If your database does not have annotations, it should return ``None``.

        **Parameters:**

        file : :py:class:`bob.bio.base.database.BioFile`
          The file for which annotations should be returned.

        **Returns:**

        annots : dict or None
          The annotations for the file, if available.
        """
        raise NotImplementedError(
            "This function must be implemented in your derived class."
        )

    #################################################################
    # Methods to provide common functionality
    #################################################################

    def model_ids(self, groups="dev"):
        """model_ids(group = 'dev') -> ids

        Returns a list of model ids for the given group, respecting the current protocol.

        **Parameters:**

        group : one of ``('dev', 'eval')``
          The group to get the model ids for.

        **Returns:**

        ids : [int] or [str]
          The list of (unique) model ids for models of the given group.
        """
        return sorted(
            self.model_ids_with_protocol(groups=groups, protocol=self.protocol)
        )

    def all_files(self, groups=None, **kwargs):
        """all_files(groups=None) -> files

        Returns all files of the database, respecting the current protocol.
        The files can be limited using the ``all_files_options`` in the constructor.

        **Parameters:**

        groups : some of ``('world', 'dev', 'eval')`` or ``None``
          The groups to get the data for.
          If ``None``, data for all groups is returned.

        kwargs: ignored

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The sorted and unique list of all files of the database.
        """
        return self.sort(
            self.objects(
                protocol=self.protocol, groups=groups, **self.all_files_options
            )
        )

    def training_files(self, step=None, arrange_by_client=False):
        """training_files(step = None, arrange_by_client = False) -> files

        Returns all training files for the given step, and arranges them by client, if desired, respecting the current protocol.
        The files for the steps can be limited using the ``..._training_options`` defined in the constructor.

        **Parameters:**

        step : one of ``('train_extractor', 'train_projector', 'train_enroller')`` or ``None``
          The step for which the training data should be returned.

        arrange_by_client : bool
          Should the training files be arranged by client?
          If set to ``True``, training files will be returned in [[:py:class:`bob.bio.base.database.BioFile`]], where each sub-list contains the files of a single client.
          Otherwise, all files will be stored in a simple [:py:class:`bob.bio.base.database.BioFile`].

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`] or [[:py:class:`bob.bio.base.database.BioFile`]]
          The (arranged) list of files used for the training of the given step.
        """
        if step is None:
            training_options = self.all_files_options
        elif step == "train_extractor":
            training_options = self.extractor_training_options
        elif step == "train_projector":
            training_options = self.projector_training_options
        elif step == "train_enroller":
            training_options = self.enroller_training_options
        else:
            raise ValueError(
                "The given step '%s' must be one of ('train_extractor', 'train_projector', 'train_enroller')"
                % step
            )

        files = self.sort(
            self.objects(
                protocol=self.protocol, groups="world", **training_options
            )
        )
        if arrange_by_client:
            return self.arrange_by_client(files)
        else:
            return files

    def test_files(self, groups=["dev"]):
        """test_files(groups = ['dev']) -> files

        Returns all test files (i.e., files used for enrollment and probing) for the given groups, respecting the current protocol.
        The files for the steps can be limited using the ``all_files_options`` defined in the constructor.

        **Parameters:**

        groups : some of ``('dev', 'eval')``
          The groups to get the data for.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The sorted and unique list of test files of the database.
        """
        return self.sort(
            self.objects(
                protocol=self.protocol, groups=groups, **self.all_files_options
            )
        )

    def enroll_files(self, model_id=None, group="dev"):
        """enroll_files(model_id, group = 'dev') -> files

        Returns a list of File objects that should be used to enroll the model with the given model id from the given group, respecting the current protocol.
        If the model_id is None (the default), enrollment files for all models are returned.

        **Parameters:**

        model_id : int or str
          A unique ID that identifies the model.

        group : one of ``('dev', 'eval')``
          The group to get the enrollment files for.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The list of files used for to enroll the model with the given model id.
        """
        if model_id:
            return self.sort(
                self.objects(
                    protocol=self.protocol,
                    groups=group,
                    model_ids=(model_id,),
                    purposes="enroll",
                    **self.all_files_options
                )
            )
        else:
            return self.sort(
                self.objects(
                    protocol=self.protocol,
                    groups=group,
                    purposes="enroll",
                    **self.all_files_options
                )
            )

    def probe_files(self, model_id=None, group="dev"):
        """probe_files(model_id = None, group = 'dev') -> files

        Returns a list of probe File objects, respecting the current protocol.
        If a ``model_id`` is specified, only the probe files that should be compared with the given model id are returned (for most databases, these are all probe files of the given group).
        Otherwise, all probe files of the given group are returned.

        **Parameters:**

        model_id : int or str or ``None``
          A unique ID that identifies the model.

        group : one of ``('dev', 'eval')``
          The group to get the enrollment files for.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The list of files used for to probe the model with the given model id.
        """
        if model_id is not None:
            files = self.objects(
                protocol=self.protocol,
                groups=group,
                model_ids=(model_id,),
                purposes="probe",
                **self.all_files_options
            )
        else:
            files = self.objects(
                protocol=self.protocol,
                groups=group,
                purposes="probe",
                **self.all_files_options
            )
        return self.sort(files)

    def object_sets(
        self,
        groups=None,
        protocol=None,
        purposes=None,
        model_ids=None,
        **kwargs
    ):
        """This function returns lists of FileSet objects, which fulfill the given restrictions.

        Keyword parameters:

        groups : str or [str]
          The groups of which the clients should be returned.
          Usually, groups are one or more elements of ('world', 'dev', 'eval')

        protocol
          The protocol for which the clients should be retrieved.
          The protocol is dependent on your database.
          If you do not have protocols defined, just ignore this field.

        purposes : str or [str]
          The purposes for which File objects should be retrieved.
          Usually, purposes are one of ('enroll', 'probe').

        model_ids : [various type]
          The model ids for which the File objects should be retrieved.
          What defines a 'model id' is dependent on the database.
          In cases, where there is only one model per client, model ids and client ids are identical.
          In cases, where there is one model per file, model ids and file ids are identical.
          But, there might also be other cases.
        """
        raise NotImplementedError(
            "This function must be implemented in your derived class."
        )

    def probe_file_sets(self, model_id=None, group="dev"):
        """probe_file_sets(model_id = None, group = 'dev') -> files

        Returns a list of probe FileSet objects, respecting the current protocol.
        If a ``model_id`` is specified, only the probe files that should be compared with the given model id are returned (for most databases, these are all probe files of the given group).
        Otherwise, all probe files of the given group are returned.

        **Parameters:**

        model_id : int or str or ``None``
          A unique ID that identifies the model.

        group : one of ``('dev', 'eval')``
          The group to get the enrollment files for.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFileSet`] or something similar
          The list of file sets used to probe the model with the given model id."""
        if model_id is not None:
            file_sets = self.object_sets(
                protocol=self.protocol,
                groups=group,
                model_ids=(model_id,),
                purposes="probe",
                **self.all_files_options
            )
        else:
            file_sets = self.object_sets(
                protocol=self.protocol,
                groups=group,
                purposes="probe",
                **self.all_files_options
            )
        return self.sort(file_sets)

    def client_id_from_model_id(self, model_id, group="dev"):
        """Return the client id associated with the given model id.
        In this base class implementation, it is assumed that only one model is enrolled for each client and, thus, client id and model id are identical.
        All key word arguments are ignored.
        Please override this function in derived class implementations to change this behavior."""
        return model_id


class ZTBioDatabase(BioDatabase):
    """This class defines another set of abstract functions that need to be implemented if your database provides the interface for computing scores used for ZT-normalization."""

    def __init__(
        self, name, z_probe_options={}, **kwargs
    ):  # Limit the z-probes
        """**Construtctor Documentation**

        This constructor tests if all implemented functions take the correct arguments.
        All keyword parameters will be passed unaltered to the :py:class:`bob.bio.base.database.BioDatabase` constructor.
        """
        # call base class constructor
        super(ZTBioDatabase, self).__init__(name, **kwargs)

        self.z_probe_options = z_probe_options

        # try if the implemented tmodel_ids_with_protocol(), tobjects() and zobjects() function have at least the required interface
        try:
            # create a value that is very unlikely a valid value for anything
            test_value = "#F9S%3*Y"
            # test if the parameters of the functions apply
            self.tmodel_ids_with_protocol(
                groups=test_value, protocol=test_value
            )
            self.tobjects(
                groups=test_value, protocol=test_value, model_ids=test_value
            )
            self.zobjects(groups=test_value, protocol=test_value)
        except TypeError as e:
            # type error indicates that the given parameters are not valid.
            raise NotImplementedError(
                str(e)
                + "\nPlease implement:\n - the tmodel_ids_with_protocol(...) function with at least the "
                "arguments 'groups' and 'protocol'\n - the tobjects(...) function with at least the arguments "
                "'groups', 'protocol' and 'model_ids'\n - the zobjects(...) function with at "
                "least the arguments 'groups' and 'protocol'"
            )
        except Exception:
            # any other error is fine at this stage.
            pass

    @abc.abstractmethod
    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        """This function returns the File objects of the T-Norm models of the given groups for the given protocol and the given model ids.

        Keyword parameters:

        groups : str or [str]
          The groups of which the model ids should be returned.
          Usually, groups are one or more elements of ('dev', 'eval')

        protocol : str
          The protocol for which the model ids should be retrieved.
          The protocol is dependent on your database.
          If you do not have protocols defined, just ignore this field.

        model_ids : [various type]
          The model ids for which the File objects should be retrieved.
          What defines a 'model id' is dependent on the database.
          In cases, where there is only one model per client, model ids and client ids are identical.
          In cases, where there is one model per file, model ids and file ids are identical.
          But, there might also be other cases.
        """
        raise NotImplementedError(
            "This function must be implemented in your derived class."
        )

    @abc.abstractmethod
    def zobjects(self, groups=None, protocol=None, **kwargs):
        """This function returns the File objects of the Z-Norm impostor files of the given groups for the given protocol.

        Keyword parameters:

        groups : str or [str]
          The groups of which the model ids should be returned.
          Usually, groups are one or more elements of ('dev', 'eval')

        protocol : str
          The protocol for which the model ids should be retrieved.
          The protocol is dependent on your database.
          If you do not have protocols defined, just ignore this field.
        """
        raise NotImplementedError(
            "This function must be implemented in your derived class."
        )

    def all_files(self, groups=["dev"], add_zt_files=True):
        """all_files(groups=None) -> files

        Returns all files of the database, including those for ZT norm, respecting the current protocol.
        The files can be limited using the ``all_files_options`` and the the ``z_probe_options`` in the constructor.

        **Parameters:**

        groups : some of ``('world', 'dev', 'eval')`` or ``None``
          The groups to get the data for.
          If ``None``, data for all groups is returned.

        add_zt_files: bool
          If set (the default), files for ZT score normalization are added.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The sorted and unique list of all files of the database.
        """
        files = self.objects(
            protocol=self.protocol, groups=groups, **self.all_files_options
        )

        # add all files that belong to the ZT-norm
        if add_zt_files and groups:
            for group in groups:
                if group == "world":
                    continue
                files += self.tobjects(
                    protocol=self.protocol, groups=group, model_ids=None
                )
                files += self.zobjects(
                    protocol=self.protocol, groups=group, **self.z_probe_options
                )
        elif add_zt_files:
            files += self.tobjects(
                protocol=self.protocol, groups=groups, model_ids=None
            )
            files += self.zobjects(
                protocol=self.protocol, groups=groups, **self.z_probe_options
            )
        return self.sort(files)

    @abc.abstractmethod
    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        """This function returns the ids of the T-Norm models of the given groups for the given protocol.

        Keyword parameters:

        groups : str or [str]
          The groups of which the model ids should be returned.
          Usually, groups are one or more elements of ('dev', 'eval')

        protocol : str
          The protocol for which the model ids should be retrieved.
          The protocol is dependent on your database.
          If you do not have protocols defined, just ignore this field.
        """
        raise NotImplementedError(
            "This function must be implemented in your derived class."
        )

    def t_model_ids(self, groups="dev"):
        """t_model_ids(group = 'dev') -> ids

        Returns a list of model ids of T-Norm models for the given group, respecting the current protocol.

        **Parameters:**

        group : one of ``('dev', 'eval')``
          The group to get the model ids for.

        **Returns:**

        ids : [int] or [str]
          The list of (unique) model ids for T-Norm models of the given group.
        """
        return sorted(
            self.tmodel_ids_with_protocol(protocol=self.protocol, groups=groups)
        )

    def t_enroll_files(self, t_model_id, group="dev"):
        """t_enroll_files(t_model_id, group = 'dev') -> files

        Returns a list of File objects that should be used to enroll the T-Norm model with the given model id from the given group, respecting the current protocol.

        **Parameters:**

        t_model_id : int or str
          A unique ID that identifies the model.

        group : one of ``('dev', 'eval')``
          The group to get the enrollment files for.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The sorted list of files used for to enroll the model with the given model id.
        """
        return self.sort(
            self.tobjects(
                protocol=self.protocol, groups=group, model_ids=(t_model_id,)
            )
        )

    def z_probe_files(self, group="dev"):
        """z_probe_files(group = 'dev') -> files

        Returns a list of probe files used to compute the Z-Norm, respecting the current protocol.
        The Z-probe files can be limited using the ``z_probe_options`` in the query to :py:meth:`bob.bio.base.database.ZTBioDatabase.z_probe_files`

        **Parameters:**

        group : one of ``('dev', 'eval')``
          The group to get the Z-norm probe files for.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFile`]
          The unique list of files used to compute the Z-norm.
        """
        return self.sort(
            self.zobjects(
                protocol=self.protocol, groups=group, **self.z_probe_options
            )
        )

    def z_probe_file_sets(self, group="dev"):
        """z_probe_file_sets(group = 'dev') -> files

        Returns a list of probe FileSet objects used to compute the Z-Norm.
        This function needs to be implemented in derived class implementations.

        **Parameters:**

        group : one of ``('dev', 'eval')``
          The group to get the Z-norm probe files for.

        **Returns:**

        files : [:py:class:`bob.bio.base.database.BioFileSet`]
          The unique list of file sets used to compute the Z-norm.
        """
        raise NotImplementedError(
            "Please implement this function in derived classes"
        )

    def client_id_from_t_model_id(self, t_model_id, group="dev"):
        """client_id_from_t_model_id(t_model_id, group = 'dev') -> client_id
        Returns the client id for the given T-Norm model id.
        In this base class implementation, we just use the :py:meth:`BioDatabase.client_id_from_model_id` function.
        Overload this function if you need another behavior.

        **Parameters:**

        t_model_id : int or str
          A unique ID that identifies the T-Norm model.
        group : one of ``('dev', 'eval')``
          The group to get the client ids for.

        **Returns:**

        client_id : [int] or [str]
          A unique ID that identifies the client, to which the T-Norm model belongs.
        """
        return self.client_id_from_model_id(t_model_id, group)
