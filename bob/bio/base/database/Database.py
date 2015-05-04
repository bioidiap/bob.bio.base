class Database:
  """This class represents the basic API for database access.
  Please use this class as a base class for your database access classes.
  Do not forget to call the constructor of this base class in your derived class."""

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
    """
    Parameters to the constructor of the Database:

    name
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

    protocol : str
      The name of the protocol that defines the default experimental setup for this database.

    training_depends_on_protocol : bool
      Specifies, if the training set used for training the extractor and the projector depend on the protocol

    models_depend_on_protocol : bool
      Specifies, if the models depend on the protocol

    kwargs
      Ignored extra arguments.
    """

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
    """This function returns a string containing all parameters of this class."""
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
    """Returns a sorted version of the given list of File's (or other structures that define an 'id' data member).
    The files will be sorted according to their id, and duplicate entries will be removed."""
    # sort files using their sort function
    sorted_files = sorted(files)
    # remove duplicates
    return [f for i,f in enumerate(sorted_files) if not i or sorted_files[i-1].id != f.id]


  def arrange_by_client(self, files):
    """Arranges the given list of files by client id.
    This function returns a list of lists of File's."""
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
    """Returns the annotations for the given File object, if available."""
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
    By default, False is returned. Overwrite the default if you need different behavior."""
    return False


  def file_names(self, files, directory, extension):
    """Returns the full path of the given File objects."""
    # return the paths of the files
    if self.uses_probe_file_sets() and files and hasattr(files[0], 'files'):
      # List of Filesets: do not remove duplicates
      return [[f.make_path(directory, extension) for f in file_set.files] for file_set in files]
    else:
      # List of files, do not remove duplicates
      return [f.make_path(directory, extension) for f in files]

  def original_file_names(self, files):
    """Returns the full path of the original data of the given File objects."""
    assert self.original_directory is not None
    assert self.original_extension is not None
    return self.file_names(files, self.original_directory, self.original_extension)


  ###########################################################################
  ### Interface functions that you need to implement in your class.
  ###########################################################################

  def all_files(self, groups = None):
    """Returns all files of the database"""
    raise NotImplementedError("Please implement this function in derived classes")


  def training_files(self, step = None, arrange_by_client = False):
    """Returns all training File objects for the given step (might be 'train_extractor', 'train_projector', 'train_enroller' or None), and arranges them by client, if desired"""
    raise NotImplementedError("Please implement this function in derived classes")


  def model_ids(self, group = 'dev'):
    """Returns a list of model ids for the given group"""
    raise NotImplementedError("Please implement this function in derived classes")


  def client_id_from_model_id(self, model_id, group = 'dev'):
    """Returns the client id for the given model id"""
    raise NotImplementedError("Please implement this function in derived classes")


  def enroll_files(self, model_id, group = 'dev'):
    """Returns a list of enrollment File objects for the given model id and the given group"""
    raise NotImplementedError("Please implement this function in derived classes")


  def probe_files(self, model_id = None, group = 'dev'):
    """Returns a list of probe File object in a specific format that should be compared with the model belonging to the given model id of the specified group"""
    raise NotImplementedError("Please implement this function in derived classes")


  def probe_file_sets(self, model_id = None, group = 'dev'):
    """Returns a list of probe FileSet object in a specific format that should be compared with the model belonging to the given model id of the specified group"""
    raise NotImplementedError("Please implement this function in derived classes")



class DatabaseZT (Database):
  """This class defines additional API functions that are required to compute ZT score normalization.
  During construction, please call the constructor of the base class 'Database' directly."""

  def t_model_ids(self, group = 'dev'):
    """Returns a list of T-Norm model ids for the given group"""
    raise NotImplementedError("Please implement this function in derived classes")

  def client_id_from_t_model_id(self, t_model_id, group = 'dev'):
    """Returns the client id for the given T-model id.
    In this base class implementation, we just use the :py:meth:`client_id_from_model_id` function.
    Overload this function if you need another behavior."""
    return self.client_id_from_model_id(t_model_id, group)

  def t_enroll_files(self, model_id, group = 'dev'):
    """Returns a list of enrollment files for the given T-Norm model id and the given group"""
    raise NotImplementedError("Please implement this function in derived classes")

  def z_probe_files(self, model_id = None, group = 'dev'):
    """Returns a list of Z-probe objects in a specific format that should be compared with the model belonging to the given model id of the specified group"""
    raise NotImplementedError("Please implement this function in derived classes")

  def z_probe_file_sets(self, model_id = None, group = 'dev'):
    """Returns a list of Z-probe FileSets object in a specific format that should be compared with the model belonging to the given model id of the specified group"""
    raise NotImplementedError("Please implement this function in derived classes")
