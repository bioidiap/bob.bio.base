#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
import logging
import os

from bob.bio.base.utils.annotations import read_annotation_file

from .. import BioFile, ZTBioDatabase
from .models import ListReader

logger = logging.getLogger("bob.bio.base")


class FileListBioDatabase(ZTBioDatabase):
    """This class provides a user-friendly interface to databases that are given as file lists.

    Parameters
    ----------

    filelists_directory : str
      The directory that contains the filelists defining the protocol(s). If you use the protocol
      attribute when querying the database, it will be appended to the base directory, such that
      several protocols are supported by the same class instance of `bob.bio.base`.

    name : str
      The name of the database

    protocol : str
      The protocol of the database. This should be a folder inside ``filelists_directory``.

    bio_file_class : ``class``
      The class that should be used to return the files.
      This can be :py:class:`bob.bio.base.database.BioFile`, :py:class:`bob.bio.spear.database.AudioBioFile`, :py:class:`bob.bio.face.database.FaceBioFile`, or anything similar.

    original_directory : str or ``None``
      The directory, where the original data can be found.

    original_extension : str or [str] or ``None``
      The filename extension of the original data, or multiple extensions.

    annotation_directory : str or ``None``
      The directory, where additional annotation files can be found.

    annotation_extension : str or ``None``
      The filename extension of the annotation files.

    annotation_type : str or ``None``
      The type of annotation that can be read.
      Currently, options are ``'eyecenter', 'named', 'idiap'``.
      See :py:func:`read_annotation_file` for details.

    dev_sub_directory : str or ``None``
      Specify a custom subdirectory for the filelists of the development set (default is ``'dev'``)

    eval_sub_directory : str or ``None``
      Specify a custom subdirectory for the filelists of the development set (default is ``'eval'``)

    world_filename : str or ``None``
      Specify a custom filename for the training filelist (default is ``'norm/train_world.lst'``)

    optional_world_1_filename : str or ``None``
      Specify a custom filename for the (first optional) training filelist
      (default is ``'norm/train_optional_world_1.lst'``)

    optional_world_2_filename : str or ``None``
      Specify a custom filename for the (second optional) training filelist
      (default is ``'norm/train_optional_world_2.lst'``)

    models_filename : str or ``None``
      Specify a custom filename for the model filelists (default is ``'for_models.lst'``)

    probes_filename : str or ``None``
      Specify a custom filename for the probes filelists (default is ``'for_probes.lst'``)

    scores_filename : str or ``None``
      Specify a custom filename for the scores filelists (default is ``'for_scores.lst'``)

    tnorm_filename : str or ``None``
      Specify a custom filename for the T-norm scores filelists (default is ``'for_tnorm.lst'``)

    znorm_filename : str or ``None``
      Specify a custom filename for the Z-norm scores filelists (default is ``'for_znorm.lst'``)

    use_dense_probe_file_list : bool or None
      Specify which list to use among ``probes_filename`` (dense) or ``scores_filename``.
      If ``None`` it is tried to be estimated based on the given parameters.

    keep_read_lists_in_memory : bool
      If set to ``True`` (the default), the lists are read only once and stored in memory.
      Otherwise the lists will be re-read for every query (not recommended).
    """

    def __init__(
        self,
        filelists_directory,
        name,
        protocol=None,
        bio_file_class=BioFile,
        original_directory=None,
        original_extension=None,
        annotation_directory=None,
        annotation_extension=".json",
        annotation_type="json",
        dev_sub_directory=None,
        eval_sub_directory=None,
        world_filename=None,
        optional_world_1_filename=None,
        optional_world_2_filename=None,
        models_filename=None,
        # For probing, use ONE of the two score file lists:
        probes_filename=None,  # File containing the probe files -> dense model/probe score matrix
        scores_filename=None,  # File containing list of model and probe files -> sparse model/probe score matrix
        # For ZT-Norm:
        tnorm_filename=None,
        znorm_filename=None,
        use_dense_probe_file_list=None,
        # if both probe_filename and scores_filename is given, what kind of list should be used?
        keep_read_lists_in_memory=True,
        # if set to True (the RECOMMENDED default) lists are read only once and stored in memory.
        **kwargs
    ):
        """Initializes the database with the file lists from the given base directory,
        and the given sub-directories and file names (which default to useful values if not given)."""

        super(FileListBioDatabase, self).__init__(
            name=name,
            protocol=protocol,
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            annotation_type=annotation_type,
            **kwargs
        )
        # extra args for pretty printing
        self._kwargs.update(
            dict(
                filelists_directory=filelists_directory,
                dev_sub_directory=dev_sub_directory,
                eval_sub_directory=eval_sub_directory,
                world_filename=world_filename,
                optional_world_1_filename=optional_world_1_filename,
                optional_world_2_filename=optional_world_2_filename,
                models_filename=models_filename,
                probes_filename=probes_filename,
                scores_filename=scores_filename,
                tnorm_filename=tnorm_filename,
                znorm_filename=znorm_filename,
                use_dense_probe_file_list=use_dense_probe_file_list,
                # if both probe_filename and scores_filename are given, what kind
                # of list should be used?
                keep_read_lists_in_memory=keep_read_lists_in_memory,
            )
        )
        # self.original_directory = original_directory
        # self.original_extension = original_extension
        self.bio_file_class = bio_file_class
        self.keep_read_lists_in_memory = keep_read_lists_in_memory
        self.list_readers = {}

        self.m_base_dir = os.path.abspath(filelists_directory)
        if not os.path.isdir(self.m_base_dir):
            raise RuntimeError(
                "Invalid directory specified %s." % (self.m_base_dir)
            )

        # sub-directories for dev and eval set:
        self.m_dev_subdir = (
            dev_sub_directory if dev_sub_directory is not None else "dev"
        )
        self.m_eval_subdir = (
            eval_sub_directory if eval_sub_directory is not None else "eval"
        )

        # training list:     format:   filename client_id
        self.m_world_filename = (
            world_filename
            if world_filename is not None
            else os.path.join("norm", "train_world.lst")
        )
        # optional training list 1:     format:   filename client_id
        self.m_optional_world_1_filename = (
            optional_world_1_filename
            if optional_world_1_filename is not None
            else os.path.join("norm", "train_optional_world_1.lst")
        )
        # optional training list 2:     format:   filename client_id
        self.m_optional_world_2_filename = (
            optional_world_2_filename
            if optional_world_2_filename is not None
            else os.path.join("norm", "train_optional_world_2.lst")
        )
        # model list:        format:   filename model_id client_id
        self.m_models_filename = (
            models_filename if models_filename is not None else "for_models.lst"
        )
        # scores list:       format:   filename model_id claimed_client_id client_id
        self.m_scores_filename = (
            scores_filename if scores_filename is not None else "for_scores.lst"
        )
        # probe list:        format:   filename client_id
        self.m_probes_filename = (
            probes_filename if probes_filename is not None else "for_probes.lst"
        )
        # T-Norm models      format:   filename model_id client_id
        self.m_tnorm_filename = (
            tnorm_filename if tnorm_filename is not None else "for_tnorm.lst"
        )
        # Z-Norm files       format:   filename client_id
        self.m_znorm_filename = (
            znorm_filename if znorm_filename is not None else "for_znorm.lst"
        )

        self.m_use_dense_probe_file_list = use_dense_probe_file_list

    def _list_reader(self, protocol):
        if protocol not in self.list_readers:
            if protocol is not None:
                protocol_dir = os.path.join(self.get_base_directory(), protocol)
                if not os.path.isdir(protocol_dir):
                    raise ValueError(
                        "The directory %s for the given protocol '%s' does not exist"
                        % (protocol_dir, protocol)
                    )
            self.list_readers[protocol] = ListReader(
                self.keep_read_lists_in_memory
            )

        return self.list_readers[protocol]

    def _make_bio(self, files):
        return [
            self.bio_file_class(
                client_id=f.client_id, path=f.path, file_id=f.id
            )
            for f in files
        ]

    def all_files(self, groups=["dev"], add_zt_files=True):
        """Returns all files for the given group. The internally stored protocol is used, throughout.

        Parameters
        ----------

        groups : [str]
          A list of groups to retrieve the files for.

        add_zt_files : bool
          If selected, also files for ZT-norm scoring will be added.
          Please select this option only if this dataset provides ZT-norm files, see :py:meth:`implements_zt`.

        Returns
        -------

        [BioFile]
          A list of all files that fulfill your query.
        """
        files = self.objects(groups, self.protocol, **self.all_files_options)
        # add all files that belong to the ZT-norm
        for group in groups:
            if group == "world":
                continue
            if add_zt_files:
                if self.implements_zt(self.protocol, group):
                    files += self.tobjects(group, self.protocol)
                    files += self.zobjects(
                        group, self.protocol, **self.z_probe_options
                    )
                else:
                    logger.warn(
                        "ZT score files are requested, but no such files are defined in group %s for protocol %s",
                        group,
                        self.protocol,
                    )

        return self.sort(self._make_bio(files))

    def groups(self, protocol=None, add_world=True, add_subworld=True):
        """This function returns the list of groups for this database.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol for which the groups should be retrieved.
          If ``None``, the internally stored protocol is used.

        add_world : bool
          Add the world groups?

        add_subworld : bool
          Add the sub-world groups? Only valid, when ``add_world=True``

        Returns
        -------

        [str]
          A list of groups
        """
        groups = []
        protocol = protocol or self.protocol
        if protocol is not None:
            if os.path.isdir(
                os.path.join(
                    self.get_base_directory(), protocol, self.m_dev_subdir
                )
            ):
                groups.append("dev")
            if os.path.isdir(
                os.path.join(
                    self.get_base_directory(), protocol, self.m_eval_subdir
                )
            ):
                groups.append("eval")
            if add_world:
                if os.path.isfile(
                    os.path.join(
                        self.get_base_directory(),
                        protocol,
                        self.m_world_filename,
                    )
                ):
                    groups.append("world")
            if add_world and add_subworld:
                if os.path.isfile(
                    os.path.join(
                        self.get_base_directory(),
                        protocol,
                        self.m_optional_world_1_filename,
                    )
                ):
                    groups.append("optional_world_1")
                if os.path.isfile(
                    os.path.join(
                        self.get_base_directory(),
                        protocol,
                        self.m_optional_world_2_filename,
                    )
                ):
                    groups.append("optional_world_2")
        else:
            if os.path.isdir(
                os.path.join(self.get_base_directory(), self.m_dev_subdir)
            ):
                groups.append("dev")
            if os.path.isdir(
                os.path.join(self.get_base_directory(), self.m_eval_subdir)
            ):
                groups.append("eval")
            if add_world:
                if os.path.isfile(
                    os.path.join(
                        self.get_base_directory(), self.m_world_filename
                    )
                ):
                    groups.append("world")
            if add_world and add_subworld:
                if os.path.isfile(
                    os.path.join(
                        self.get_base_directory(),
                        self.m_optional_world_1_filename,
                    )
                ):
                    groups.append("optional_world_1")
                if os.path.isfile(
                    os.path.join(
                        self.get_base_directory(),
                        self.m_optional_world_2_filename,
                    )
                ):
                    groups.append("optional_world_2")
        return groups

    def implements_zt(self, protocol=None, groups=None):
        """Checks if the file lists for the ZT score normalization are available.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol for which the groups should be retrieved.

        groups : str or [str] or ``None``
          The groups for which the ZT score normalization file lists should be checked ``('dev', 'eval')``.

        Returns
        -------

        bool
          ``True`` if the all file lists for ZT score normalization exist, otherwise ``False``.
        """
        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups, "group", self.groups(protocol, add_world=False)
        )

        for group in groups:
            for t in ["for_tnorm", "for_znorm"]:
                if not os.path.exists(self._get_list_file(group, t, protocol)):
                    return False
        # all files exist
        return True

    def uses_dense_probe_file(self, protocol):
        """Determines if a dense probe file list is used based on the existence of parameters."""
        # return, whatever was specified in constructor, if not None
        if self.m_use_dense_probe_file_list is not None:
            return self.m_use_dense_probe_file_list

        # check the existence of the files
        probes = True
        scores = True
        for group in self.groups(protocol, add_world=False):
            probes = probes and os.path.exists(
                self._get_list_file(group, type="for_probes", protocol=protocol)
            )
            scores = scores and os.path.exists(
                self._get_list_file(group, type="for_scores", protocol=protocol)
            )
        # decide, which score files are available
        if probes and not scores:
            return True
        if not probes and scores:
            return False
        raise ValueError(
            "Unable to determine, which way of probing should be used. Please specify."
        )

    def get_base_directory(self):
        """Returns the base directory where the filelists defining the database
        are located."""
        return self.m_base_dir

    def set_base_directory(self, filelists_directory):
        """Resets the base directory where the filelists defining the database
        are located."""
        self.m_base_dir = filelists_directory
        if not os.path.isdir(self.filelists_directory):
            raise RuntimeError(
                "Invalid directory specified %s." % (self.filelists_directory)
            )

    def _get_list_file(self, group, type=None, protocol=None):
        if protocol:
            base_directory = os.path.join(self.get_base_directory(), protocol)
        else:
            base_directory = self.get_base_directory()
        if group == "world":
            return os.path.join(base_directory, self.m_world_filename)
        elif group == "optional_world_1":
            return os.path.join(
                base_directory, self.m_optional_world_1_filename
            )
        elif group == "optional_world_2":
            return os.path.join(
                base_directory, self.m_optional_world_2_filename
            )
        else:
            group_dir = (
                self.m_dev_subdir if group == "dev" else self.m_eval_subdir
            )
            list_name = {
                "for_models": self.m_models_filename,
                "for_probes": self.m_probes_filename,
                "for_scores": self.m_scores_filename,
                "for_tnorm": self.m_tnorm_filename,
                "for_znorm": self.m_znorm_filename,
            }[type]
            return os.path.join(base_directory, group_dir, list_name)

    def client_id_from_model_id(self, model_id, group="dev"):
        """Returns the client id that is connected to the given model id.

        Parameters
        ----------

        model_id : str or ``None``
          The model id for which the client id should be returned.

        groups : str or [str] or ``None``
          (optional) the groups, the client belongs to.
          Might be one or more of ``('dev', 'eval', 'world', 'optional_world_1', 'optional_world_2')``.
          If groups are given, only these groups are considered.

        protocol : str or ``None``
          The protocol to consider.

        Returns
        -------

        str
          The client id for the given model id, if found.
        """
        protocol = self.protocol
        groups = self.check_parameters_for_validity(
            group,
            "group",
            self.groups(protocol),
            default_parameters=self.groups(protocol, add_subworld=False),
        )

        for group in groups:
            model_dict = self._list_reader(protocol).read_models(
                self._get_list_file(group, "for_models", protocol),
                group,
                "for_models",
            )
            if model_id in model_dict:
                return model_dict[model_id]

        raise ValueError(
            "The given model id '%s' cannot be found in one of the groups '%s'"
            % (model_id, groups)
        )

    def client_id_from_t_model_id(self, t_model_id, group="dev"):
        """Returns the client id that is connected to the given T-Norm model id.

        Parameters
        ----------

        model_id : str or ``None``
          The model id for which the client id should be returned.

        groups : str or [str] or ``None``
          (optional) the groups, the client belongs to.
          Might be one or more of ``('dev', 'eval')``.
          If groups are given, only these groups are considered.

        Returns
        -------

        str
          The client id for the given model id of a T-Norm model, if found.
        """
        protocol = self.protocol
        groups = self.check_parameters_for_validity(
            group, "group", self.groups(protocol, add_world=False)
        )

        for group in groups:
            model_dict = self._list_reader(protocol).read_models(
                self._get_list_file(group, "for_tnorm", protocol),
                group,
                "for_tnorm",
            )
            if t_model_id in model_dict:
                return model_dict[t_model_id]

        raise ValueError(
            "The given T-norm model id '%s' cannot be found in one of the groups '%s'"
            % (t_model_id, groups)
        )

    def __client_id_list__(self, groups, type, protocol=None):
        ids = set()
        protocol = protocol or self.protocol
        # read all lists for all groups and extract the model ids
        for group in groups:
            files = self._list_reader(protocol).read_list(
                self._get_list_file(group, type, protocol), group, type
            )
            for file in files:
                ids.add(file.client_id)
        return ids

    def client_ids(self, protocol=None, groups=None):
        """Returns a list of client ids for the specific query by the user.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        groups : str or [str] or ``None``
          The groups to which the clients belong ``('dev', 'eval', 'world', 'optional_world_1', 'optional_world_2')``.

        Returns
        -------

        [str]
          A list containing all the client ids which have the given properties.
        """

        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups,
            "group",
            self.groups(protocol),
            default_parameters=self.groups(protocol, add_subworld=False),
        )

        return self.__client_id_list__(groups, "for_models", protocol)

    def tclient_ids(self, protocol=None, groups=None):
        """Returns a list of T-Norm client ids for the specific query by the user.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        groups : str or [str] or ``None``
          The groups to which the clients belong ("dev", "eval").

        Returns
        -------

        [str]
          A list containing all the T-Norm client ids which have the given properties.
        """

        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups, "group", self.groups(protocol, add_world=False)
        )

        return self.__client_id_list__(groups, "for_tnorm", protocol)

    def zclient_ids(self, protocol=None, groups=None):
        """Returns a list of Z-Norm client ids for the specific query by the user.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        groups : str or [str] or ``None``
          The groups to which the clients belong ("dev", "eval").

        Returns
        -------

        [str]
          A list containing all the Z-Norm client ids which have the given properties.
        """

        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups, "group", self.groups(protocol, add_world=False)
        )

        return self.__client_id_list__(groups, "for_znorm", protocol)

    def __model_id_list__(self, groups, type, protocol=None):
        ids = set()
        protocol = protocol or self.protocol
        # read all lists for all groups and extract the model ids
        for group in groups:
            dict = self._list_reader(protocol).read_models(
                self._get_list_file(group, type, protocol), group, type
            )
            ids.update(dict.keys())
        return list(ids)

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        """Returns a list of model ids for the specific query by the user.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        groups : str or [str] or ``None``
          The groups to which the models belong ``('dev', 'eval', 'world', 'optional_world_1', 'optional_world_2')``.

        Returns
        -------

        [str]
          A list containing all the model ids which have the given properties.
        """
        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups, "group", self.groups(protocol=protocol)
        )

        return self.__model_id_list__(groups, "for_models", protocol)

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        """Returns a list of T-Norm model ids for the specific query by the user.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        groups : str or [str] or ``None``
          The groups to which the models belong ``('dev', 'eval')``.

        Returns
        -------

        [str]
          A list containing all the T-Norm model ids belonging to the given group.
        """
        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups, "group", self.groups(protocol, add_world=False)
        )

        return self.__model_id_list__(groups, "for_tnorm", protocol)

    def objects(
        self,
        groups=None,
        protocol=None,
        purposes=None,
        model_ids=None,
        classes=None,
        **kwargs
    ):
        """Returns a set of :py:class:`bob.bio.base.database.BioFile` objects for the specific query by the user.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        purposes : str or [str] or ``None``
          The purposes required to be retrieved ``('enroll', 'probe')`` or a tuple
          with several of them. If ``None`` is given (this is the default), it is
          considered the same as a tuple with all possible values. This field is
          ignored for the data from the ``'world', 'optional_world_1', 'optional_world_2'`` groups.

        model_ids : str or [str] or ``None``
          Only retrieves the files for the provided list of model ids (claimed
          client id). If ``None`` is given (this is the default), no filter over
          the model_ids is performed.

        groups : str or [str] or ``None``
          One of the groups ``('dev', 'eval', 'world', 'optional_world_1', 'optional_world_2')`` or a tuple with several of them.
          If ``None`` is given (this is the default), it is considered to be the existing subset of ``('world', 'dev', 'eval')``.

        classes : str or [str] or ``None``
          The classes (types of accesses) to be retrieved ``('client', 'impostor')``
          or a tuple with several of them. If ``None`` is given (this is the
          default), it is considered the same as a tuple with all possible values.

          .. note::
             Classes are not allowed to be specified when 'probes_filename' is used in the constructor.

        Returns
        -------

        [BioFile]
          A list of :py:class:`BioFile` objects considering all the filtering criteria.
        """

        protocol = protocol or self.protocol
        if self.uses_dense_probe_file(protocol) and classes is not None:
            raise ValueError(
                "To be able to use the 'classes' keyword, please use the 'for_scores.lst' list file."
            )

        purposes = self.check_parameters_for_validity(
            purposes, "purpose", ("enroll", "probe")
        )
        groups = self.check_parameters_for_validity(
            groups,
            "group",
            self.groups(protocol),
            default_parameters=self.groups(protocol, add_subworld=False),
        )
        classes = self.check_parameters_for_validity(
            classes, "class", ("client", "impostor")
        )

        if isinstance(model_ids, str):
            model_ids = (model_ids,)

        # first, collect all the lists that we want to process
        lists = []
        probe_lists = []
        if "world" in groups:
            lists.append(
                self._list_reader(protocol).read_list(
                    self._get_list_file("world", protocol=protocol), "world"
                )
            )
        if "optional_world_1" in groups:
            lists.append(
                self._list_reader(protocol).read_list(
                    self._get_list_file("optional_world_1", protocol=protocol),
                    "optional_world_1",
                )
            )
        if "optional_world_2" in groups:
            lists.append(
                self._list_reader(protocol).read_list(
                    self._get_list_file("optional_world_2", protocol=protocol),
                    "optional_world_2",
                )
            )

        for group in ("dev", "eval"):
            if group in groups:
                if "enroll" in purposes:
                    lists.append(
                        self._list_reader(protocol).read_list(
                            self._get_list_file(
                                group, "for_models", protocol=protocol
                            ),
                            group,
                            "for_models",
                        )
                    )
                if "probe" in purposes:
                    if self.uses_dense_probe_file(protocol):
                        probe_lists.append(
                            self._list_reader(protocol).read_list(
                                self._get_list_file(
                                    group, "for_probes", protocol=protocol
                                ),
                                group,
                                "for_probes",
                            )
                        )
                    else:
                        probe_lists.append(
                            self._list_reader(protocol).read_list(
                                self._get_list_file(
                                    group, "for_scores", protocol=protocol
                                ),
                                group,
                                "for_scores",
                            )
                        )

        # now, go through the lists and filter the elements

        # remember the file ids that are already in the list
        file_ids = set()
        retval = []

        # non-probe files; just filter by model id
        for list in lists:
            for file in list:
                # check if we already have this file
                if file.id not in file_ids:
                    if model_ids is None or file._model_id in model_ids:
                        file_ids.add(file.id)
                        retval.append(file)

        # probe files; filter by model id and by class
        for list in probe_lists:
            if self.uses_dense_probe_file(protocol):
                # dense probing is used; do not filter over the model ids and not over the classes
                # -> just add all probe files
                for file in list:
                    if file.id not in file_ids:
                        file_ids.add(file.id)
                        retval.append(file)

            else:
                # sparse probing is used; filter over model ids and over the classes
                for file in list:
                    # filter by model id
                    if model_ids is None or file._model_id in model_ids:
                        # filter by class
                        if (
                            "client" in classes
                            and file.client_id == file.claimed_id
                        ) or (
                            "impostor" in classes
                            and file.client_id != file.claimed_id
                        ):
                            # check if we already have this file
                            if file.id not in file_ids:
                                file_ids.add(file.id)
                                retval.append(file)

        return self._make_bio(retval)

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        """Returns a list of :py:class:`bob.bio.base.database.BioFile` objects for enrolling T-norm models for score normalization.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        model_ids : str or [str] or ``None``
          Only retrieves the files for the provided list of model ids (claimed
          client id). If ``None`` is given (this is the default), no filter over
          the model_ids is performed.

        groups : str or [str] or ``None``
          The groups to which the models belong ``('dev', 'eval')``.

        Returns
        -------

        [BioFile]
          A list of :py:class:`BioFile` objects considering all the filtering criteria.
        """
        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups, "group", self.groups(protocol, add_world=False)
        )

        if isinstance(model_ids, str):
            model_ids = (model_ids,)

        # iterate over the lists and extract the files
        # we assume that there is no duplicate file here...
        retval = []
        for group in groups:
            for file in self._list_reader(protocol).read_list(
                self._get_list_file(group, "for_tnorm", protocol),
                group,
                "for_tnorm",
            ):
                if model_ids is None or file._model_id in model_ids:
                    retval.append(file)

        return self._make_bio(retval)

    def zobjects(self, groups=None, protocol=None, **kwargs):
        """Returns a list of :py:class:`BioFile` objects to perform Z-norm score normalization.

        Parameters
        ----------

        protocol : str or ``None``
          The protocol to consider

        groups : str or [str] or ``None``
          The groups to which the clients belong ``('dev', 'eval')``.

        Returns
        -------

        [BioFile]
          A list of File objects considering all the filtering criteria.
        """

        protocol = protocol or self.protocol
        groups = self.check_parameters_for_validity(
            groups, "group", self.groups(protocol, add_world=False)
        )

        # iterate over the lists and extract the files
        # we assume that there is no duplicate file here...
        retval = []
        for group in groups:
            retval.extend(
                [
                    file
                    for file in self._list_reader(protocol).read_list(
                        self._get_list_file(group, "for_znorm", protocol),
                        group,
                        "for_znorm",
                    )
                ]
            )

        return self._make_bio(retval)

    def annotations(self, file):
        """Reads the annotations for the given file id from file and returns them in a dictionary.

        Parameters
        ----------

        file : BioFile
          The BioFile object for which the annotations should be read.

        Returns
        -------

        dict
          The annotations as a dictionary, e.g.: ``{'reye':(re_y,re_x), 'leye':(le_y,le_x)}``
        """
        if self.annotation_directory is None:
            return None

        # since the file id is equal to the file name, we can simply use it
        annotation_file = os.path.join(
            self.annotation_directory, file.id + self.annotation_extension
        )

        # return the annotations as read from file
        return read_annotation_file(annotation_file, self.annotation_type)

    def original_file_name(self, file, check_existence=True):
        """Returns the original file name of the given file.

        This interface supports several original extensions, so that file lists can contain images
        of different data types.

        When multiple original extensions are specified, this function will check the existence of any of
        these file names, and return the first one that actually exists.
        In this case, the ``check_existence`` flag is ignored.

        Parameters
        ----------

        file : BioFile
          The BioFile object for which the file name should be returned.

        check_existence : bool
          Should the existence of the original file be checked?
          (Ignored when multiple original extensions were specified in the constructor.)

        Returns
        -------

        str
          The full path of the original data file.
        """

        if isinstance(self.original_extension, str):
            # extract file name
            file_name = file.make_path(
                self.original_directory, self.original_extension
            )
            if not check_existence or os.path.exists(file_name):
                return file_name

        # check all registered extensions
        for extension in self.original_extension:
            file_name = file.make_path(self.original_directory, extension)
            if os.path.exists(file_name):
                return file_name

        # None of the extensions matched
        raise IOError(
            "File '%s' does not exist with any of the extensions '%s'"
            % (
                file.make_path(self.original_directory, None),
                self.original_extension,
            )
        )
