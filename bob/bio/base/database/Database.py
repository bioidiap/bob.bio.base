class Database:
  """This class represents the basic API for database access.
  Please use this class as a base class for your database access classes.
  Do not forget to call the constructor of this base class in your derived class.

  **Parameters:**

  name : str
    A unique name for the database.

  original_directory : str
    The directory where the original data of the database are stored.

  original_extension : str
    The file name extension of the original data.

  annotation_directory : str
    The directory where the image annotations of the database are stored, if any.

  annotation_extension : str
    The file name extension of the annotation files.

  annotation_type : str
    The type of the annotation file to read, see :py:func:`bob.db.verification.utils.read_annotation_file` for accepted formats.

  protocol : str or ``None``
    The name of the protocol that defines the default experimental setup for this database.

    .. todo:: Check if the ``None`` protocol is supported.

  training_depends_on_protocol : bool
    Specifies, if the training set used for training the extractor and the projector depend on the protocol.
    This flag is used to avoid re-computation of data when running on the different protocols of the same database.

  models_depend_on_protocol : bool
    Specifies, if the models depend on the protocol.
    This flag is used to avoid re-computation of models when running on the different protocols of the same database.

  kwargs
    Ignored extra arguments.
  """

  def __init__(
     self,
     name,
     original_directory = None,
     original_extension = None,
     annotation_directory = None,
     annotation_extension = '.pos',
     annotation_type = None,
     protocol = 'Default',
     training_depends_on_protocol = False,
     models_depend_on_protocol = False,
     **kwargs
  ):
    assert isinstance(name, str)

    self.name = name
    self.original_directory = original_directory
    self.original_extension = original_extension
    self.annotation_directory = annotation_directory
    self.annotation_extension = annotation_extension
    self.annotation_type = annotation_type
    self.protocol = protocol
    self.training_depends_on_protocol = training_depends_on_protocol
    self.models_depend_on_protocol = models_depend_on_protocol


  def __str__(self):
    """__str__() -> info

    This function returns all parameters of this class.

    **Returns:**

    info : str
      A string containing the full information of all parameters of this class.
    """
    params = "name=%s, protocol=%s, original_directory=%s, original_extension=%s" % (self.name, self.protocol, self.original_directory, self.original_extension)
    if self.annotation_type is not None:
      params += ", annotation_type=%s" % annotation_type
      if self.annotation_directory: params += ", annotation_directory=%s" % self.annotation_directory
      params += ", annotation_extension=%s" % self.annotation_extension
    params += ", training_depends_on_protocol=%s, models_depend_on_protocol=%s" % (self.training_depends_on_protocol, self.models_depend_on_protocol)
    return "%s(%s)" % (str(self.__class__), params)


  ###########################################################################
  ### Helper functions that you might want to use in derived classes
  ###########################################################################
  def sort(self, files):
    """sort(files) -> sorted

    Returns a sorted version of the given list of File's (or other structures that define an 'id' data member).
    The files will be sorted according to their id, and duplicate entries will be removed.

    **Parameters:**

    files : [:py:class:`File`]
      The list of files to be uniquified and sorted.

    **Returns:**

    sorted : [:py:class:`File`]
      The sorted list of files, with duplicate :py:attr:`File.id`\s being removed.
    """
    # sort files using their sort function
    sorted_files = sorted(files)
    # remove duplicates
    return [f for i,f in enumerate(sorted_files) if not i or sorted_files[i-1].id != f.id]


  def arrange_by_client(self, files):
    """arrange_by_client(files) -> files_by_client

    Arranges the given list of files by client id.
    This function returns a list of lists of File's.

    **Parameters:**

    files : :py:class:`File`
      A list of files that should be split up by :py:attr:`File.client_id`.

    **Returns:**

    files_by_client : [[:py:class:`File`]]
      The list of lists of files, where each sub-list groups the files with the same :py:attr:`File.client_id`
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


  def annotations(self, file):
    """annotations(file) -> annots

    Returns the annotations for the given File object, if available.
    It uses :py:func:`bob.db.verification.utils.read_annotation_file` to load the annotations.

    **Parameters:**

    file : :py:class:`File`
      The file for which annotations should be returned.

    **Returns:**

    annots : dict or None
      The annotations for the file, if available.
    """
    if self.annotation_directory:
      try:
        import bob.db.verification.utils
        annotation_path = os.path.join(self.annotation_directory, file.path + self.annotation_extension)
        return bob.db.verification.utils.read_annotation_file(annotation_path, self.annotation_type)
      except ImportError as e:
        from .. import utils
        utils.error("Cannot import bob.db.verification.utils: '%s'. No annotation is read." % e)

    return None


  def uses_probe_file_sets(self):
    """Defines if, for the current protocol, the database uses several probe files to generate a score.
    By default, ``False`` is returned. Overwrite the default if you need different behavior."""
    return False


  def file_names(self, files, directory, extension):
    """file_names(files, directory, extension) -> paths

    Returns the full path of the given File objects.

    **Parameters:**

    files : [:py:class:`File`]
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
    if self.uses_probe_file_sets() and files and hasattr(files[0], 'files'):
      # List of Filesets: do not remove duplicates
      return [[f.make_path(directory, extension) for f in file_set.files] for file_set in files]
    else:
      # List of files, do not remove duplicates
      return [f.make_path(directory, extension) for f in files]

  def original_file_names(self, files):
    """original_file_names(files) -> paths

    Returns the full path of the original data of the given File objects.

    **Parameters:**

    files : [:py:class:`File`]
      The list of file object to retrieve the original data file names for.

    **Returns:**

    paths : [str] or [[str]]
      The paths extracted for the files, in the same order.
      If this database provides file sets, a list of lists of file names is returned, one sub-list for each file set.
    """
    assert self.original_directory is not None
    assert self.original_extension is not None
    return self.file_names(files, self.original_directory, self.original_extension)


  ###########################################################################
  ### Interface functions that you need to implement in your class.
  ###########################################################################

  def all_files(self, groups = None):
    """all_files(groups=None) -> files

    Returns all files of the database.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    groups : some of ``('world', 'dev', 'eval')`` or ``None``
      The groups to get the data for.
      If ``None``, data for all groups is returned.

    **Returns:**

    files : [:py:class:`File`]
      The sorted and unique list of all files of the database.
    """
    raise NotImplementedError("Please implement this function in derived classes")


  def training_files(self, step = None, arrange_by_client = False):
    """training_files(step = None, arrange_by_client = False) -> files

    Returns all training File objects for the given step, and arranges them by client, if desired.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    step : one of ``('train_extractor', 'train_projector', 'train_enroller')`` or ``None``
      The step for which the training data should be returned.
      Might be ignored in derived class implementations.

    arrange_by_client : bool
      Should the training files be arranged by client?

      .. note::
         You can use :py:meth:`arrange_by_client` in derived class implementations to arrange the files.

    **Returns:**

    files : [:py:class:`File`] or [[:py:class:`File`]]
      The (arranged) list of files used for the training of the given step.
    """
    raise NotImplementedError("Please implement this function in derived classes")


  def model_ids(self, group = 'dev'):
    """model_ids(group = 'dev') -> ids

    Returns a list of model ids for the given group.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the model ids for.

    **Returns:**

    ids : [int] or [str]
      The list of (unique) model ids for the given group.
    """
    raise NotImplementedError("Please implement this function in derived classes")


  def client_id_from_model_id(self, model_id, group = 'dev'):
    """client_id_from_model_id(model_id, group = 'dev') -> client_id

    In some databases, each client can contain several models.
    Hence, client and model ids differ.
    This function converts the given model id into its according the client id.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    model_id : int or str
      A unique ID that identifies the model for the client.

    group : one of ``('dev', 'eval')``
      The group to get the client ids for.

    **Returns:**

    client_id : [int] or [str]
      A unique ID that identifies the client, to which the model belongs.
    """
    raise NotImplementedError("Please implement this function in derived classes")


  def enroll_files(self, model_id, group = 'dev'):
    """enroll_files(model_id, group = 'dev') -> files

    Returns a list of File objects that should be used to enroll the model with the given model id from the given group.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    model_id : int or str
      A unique ID that identifies the model.

    group : one of ``('dev', 'eval')``
      The group to get the enrollment files for.

    **Returns:**

    files : [:py:class:`File`]
      The list of files used for to enroll the model with the given model id.
    """
    raise NotImplementedError("Please implement this function in derived classes")


  def probe_files(self, model_id = None, group = 'dev'):
    """probe_files(model_id = None, group = 'dev') -> files

    Returns a list of probe File objects.
    If a ``model_id`` is specified, only the probe files that should be compared with the given model id are returned (for most databases, these are all probe files of the given group).
    Otherwise, all probe files of the given group are returned.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    model_id : int or str or ``None``
      A unique ID that identifies the model.

    group : one of ``('dev', 'eval')``
      The group to get the enrollment files for.

    **Returns:**

    files : [:py:class:`File`]
      The list of files used for to probe the model with the given model id.
    """
    raise NotImplementedError("Please implement this function in derived classes")


  def probe_file_sets(self, model_id = None, group = 'dev'):
    """probe_file_sets(model_id = None, group = 'dev') -> files

    Returns a list of probe FileSet objects.
    If a ``model_id`` is specified, only the probe files that should be compared with the given model id are returned (for most databases, these are all probe files of the given group).
    Otherwise, all probe files of the given group are returned.
    This function needs to be implemented in derived class implementations, if the :py:meth:`uses_probe_file_sets` returns ``True``.

    **Parameters:**

    model_id : int or str or ``None``
      A unique ID that identifies the model.

    group : one of ``('dev', 'eval')``
      The group to get the enrollment files for.

    **Returns:**

    files : [:py:class:`FileSet`]
      The list of file sets used to probe the model with the given model id."""
    raise NotImplementedError("Please implement this function in derived classes")



class DatabaseZT (Database):
  """This class defines additional API functions that are required to compute ZT score normalization.
  This class does not define a constructor.
  During construction of a derived class, please call the constructor of the base class :py:class:`Database` directly."""

  def t_model_ids(self, group = 'dev'):
    """t_model_ids(group = 'dev') -> ids

    Returns a list of model ids of T-Norm models for the given group.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the model ids for.

    **Returns:**

    ids : [int] or [str]
      The list of (unique) model ids for T-Norm models of the given group.
    """
    raise NotImplementedError("Please implement this function in derived classes")


  def client_id_from_t_model_id(self, t_model_id, group = 'dev'):
    """client_id_from_t_model_id(t_model_id, group = 'dev') -> client_id

    Returns the client id for the given T-Norm model id.
    In this base class implementation, we just use the :py:meth:`client_id_from_model_id` function.
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

  def t_enroll_files(self, t_model_id, group = 'dev'):
    """t_enroll_files(t_model_id, group = 'dev') -> files

    Returns a list of File objects that should be used to enroll the T-Norm model with the given model id from the given group.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    t_model_id : int or str
      A unique ID that identifies the model.

    group : one of ``('dev', 'eval')``
      The group to get the enrollment files for.

    **Returns:**

    files : [:py:class:`File`]
      The list of files used for to enroll the model with the given model id.
    """
    raise NotImplementedError("Please implement this function in derived classes")

  def z_probe_files(self, group = 'dev'):
    """z_probe_files(group = 'dev') -> files

    Returns a list of probe File objects used to compute the Z-Norm.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the Z-norm probe files for.

    **Returns:**

    files : [:py:class:`File`]
      The unique list of files used to compute the Z-norm.
    """
    raise NotImplementedError("Please implement this function in derived classes")

  def z_probe_file_sets(self, group = 'dev'):
    """z_probe_file_sets(group = 'dev') -> files

    Returns a list of probe FileSet objects used to compute the Z-Norm.
    This function needs to be implemented in derived class implementations.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the Z-norm probe files for.

    **Returns:**

    files : [:py:class:`FileSet`]
      The unique list of file sets used to compute the Z-norm.
    """
    raise NotImplementedError("Please implement this function in derived classes")
