from .Database import Database, DatabaseZT
import os

import bob.db.verification.utils

class DatabaseBob (Database):
  """This class can be used whenever you have a database that follows the Bob verification database interface, which is defined in :py:class:`bob.db.verification.utils.Database`

  **Parameters:**

  database : derivative of :py:class:`bob.db.verification.utils.Database`
    The database instance (such as a :py:class:`bob.db.atnt.Database`) that provides the actual interface, see :ref:`verification_databases` for a list.

  all_files_options : dict
    Dictionary of options passed to the :py:meth:`bob.db.verification.utils.Database.objects` database query when retrieving all data.

  extractor_training_options : dict
    Dictionary of options passed to the :py:meth:`bob.db.verification.utils.Database.objects` database query used to retrieve the files for the extractor training.

  projector_training_options : dict
    Dictionary of options passed to the :py:meth:`bob.db.verification.utils.Database.objects` database query used to retrieve the files for the projector training.

  enroller_training_options : dict
    Dictionary of options passed to the :py:meth:`bob.db.verification.utils.Database.objects` database query used to retrieve the files for the enroller training.

  check_original_files_for_existence : bool
    Enables to test for the original data files when querying the database.

  kwargs : ``key=value`` pairs
    The arguments of the :py:class:`Database` base class constructor.

    .. note:: Usually, the ``name``, ``protocol``, ``training_depends_on_protocol`` and ``models_depend_on_protocol`` keyword parameters of the base class constructor need to be specified.
  """

  def __init__(
      self,
      database,  # The bob database that is used
      all_files_options = {}, # additional options for the database query that can be used to extract all files
      extractor_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
      projector_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
      enroller_training_options = {},  # additional options for the database query that can be used to extract the training files for the extractor training
      check_original_files_for_existence = False,
      **kwargs  # The default parameters of the base class
  ):

    Database.__init__(
        self,
        **kwargs
    )

    assert isinstance(database, bob.db.verification.utils.Database), "Only databases derived from bob.db.verification.utils.Database are supported by this interface. Please implement your own bob.bio.base.database.Database interface."

    self.database = database
    self.original_directory = database.original_directory
    try:
      self.annotation_directory = database.annotation_directory
    except AttributeError:
      pass

    self.all_files_options = all_files_options
    self.extractor_training_options = extractor_training_options
    self.projector_training_options = projector_training_options
    self.enroller_training_options = enroller_training_options
    self.check_existence = check_original_files_for_existence

    self._kwargs = kwargs


  def __str__(self):
    """__str__() -> info

    This function returns all parameters of this class (and its derived class).

    **Returns:**

    info : str
      A string containing the full information of all parameters of this (and the derived) class.
    """
    params = ", ".join(["%s=%s" % (key, value) for key, value in self._kwargs.items()])
    params += ", original_directory=%s, original_extension=%s" % (self.original_directory, self.original_extension)
    if self.all_files_options: params += ", all_files_options=%s"%self.all_files_options
    if self.extractor_training_options: params += ", extractor_training_options=%s"%self.extractor_training_options
    if self.projector_training_options: params += ", projector_training_options=%s"%self.projector_training_options
    if self.enroller_training_options: params += ", enroller_training_options=%s"%self.enroller_training_options

    return "%s(%s)" % (str(self.__class__), params)


  def replace_directories(self, replacements = None):
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
      self.database.original_directory = replacements[self.database.original_directory]

    try:
      if self.annotation_directory in replacements:
        self.annotation_directory = replacements[self.annotation_directory]
        self.database.annotation_directory = replacements[self.database.annotation_directory]
    except AttributeError:
      pass


  def uses_probe_file_sets(self):
    """Defines if, for the current protocol, the database uses several probe files to generate a score."""
    return self.database.provides_file_set_for_protocol(self.protocol)


  def all_files(self, groups = None):
    """all_files(groups=None) -> files

    Returns all files of the database, respecting the current protocol.
    The files can be limited using the ``all_files_options`` in the constructor.

    **Parameters:**

    groups : some of ``('world', 'dev', 'eval')`` or ``None``
      The groups to get the data for.
      If ``None``, data for all groups is returned.

    **Returns:**

    files : [:py:class:`bob.db.verification.utils.File`]
      The sorted and unique list of all files of the database.
    """
    return self.sort(self.database.objects(protocol = self.protocol, groups = groups, **self.all_files_options))


  def training_files(self, step = None, arrange_by_client = False):
    """training_files(step = None, arrange_by_client = False) -> files

    Returns all training files for the given step, and arranges them by client, if desired, respecting the current protocol.
    The files for the steps can be limited using the ``..._training_options`` defined in the constructor.

    **Parameters:**

    step : one of ``('train_extractor', 'train_projector', 'train_enroller')`` or ``None``
      The step for which the training data should be returned.

    arrange_by_client : bool
      Should the training files be arranged by client?
      If set to ``True``, training files will be returned in [[:py:class:`bob.db.verification.utils.File`]], where each sub-list contains the files of a single client.
      Otherwise, all files will be stored in a simple [:py:class:`bob.db.verification.utils.File`].

    **Returns:**

    files : [:py:class:`bob.db.verification.utils.File`] or [[:py:class:`bob.db.verification.utils.File`]]
      The (arranged) list of files used for the training of the given step.
    """
    if step is None:
      training_options = self.all_files_options
    elif step == 'train_extractor':
      training_options = self.extractor_training_options
    elif step == 'train_projector':
      training_options = self.projector_training_options
    elif step == 'train_enroller':
      training_options = self.enroller_training_options
    else:
      raise ValueError("The given step '%s' must be one of ('train_extractor', 'train_projector', 'train_enroller')" % step)

    files = self.sort(self.database.objects(protocol = self.protocol, groups = 'world', **training_options))
    if arrange_by_client:
      return self.arrange_by_client(files)
    else:
      return files

  def test_files(self, groups = ['dev']):
    """test_files(groups = ['dev']) -> files

    Returns all test files (i.e., files used for enrollment and probing) for the given groups, respecting the current protocol.
    The files for the steps can be limited using the ``all_files_options`` defined in the constructor.

    **Parameters:**

    groups : some of ``('dev', 'eval')``
      The groups to get the data for.

    **Returns:**

    files : [:py:class:`bob.db.verification.utils.File`]
      The sorted and unique list of test files of the database.
    """
    return self.sort(self.database.test_files(protocol = self.protocol, groups = groups, **self.all_files_options))

  def model_ids(self, group = 'dev'):
    """model_ids(group = 'dev') -> ids

    Returns a list of model ids for the given group, respecting the current protocol.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the model ids for.

    **Returns:**

    ids : [int] or [str]
      The list of (unique) model ids for the given group.
    """
    return sorted(self.database.model_ids(protocol = self.protocol, groups = group))


  def client_id_from_model_id(self, model_id, group = 'dev'):
    """client_id_from_model_id(model_id, group = 'dev') -> client_id

    Uses :py:meth:`bob.db.verification.utils.Database.get_client_id_from_model_id` to retrieve the client id for the given model id.

    **Parameters:**

    model_id : int or str
      A unique ID that identifies the model for the client.

    group : one of ``('dev', 'eval')``
      The group to get the client ids for.

    **Returns:**

    client_id : [int] or [str]
      A unique ID that identifies the client, to which the model belongs.
    """
    return self.database.get_client_id_from_model_id(model_id)


  def enroll_files(self, model_id, group = 'dev'):
    """enroll_files(model_id, group = 'dev') -> files

    Returns a list of File objects that should be used to enroll the model with the given model id from the given group, respecting the current protocol.

    **Parameters:**

    model_id : int or str
      A unique ID that identifies the model.

    group : one of ``('dev', 'eval')``
      The group to get the enrollment files for.

    **Returns:**

    files : [:py:class:`bob.db.verification.utils.File`]
      The list of files used for to enroll the model with the given model id.
    """
    return self.sort(self.database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'enroll', **self.all_files_options))


  def probe_files(self, model_id = None, group = 'dev'):
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

    files : [:py:class:`bob.db.verification.utils.File`]
      The list of files used for to probe the model with the given model id.
    """
    if model_id is not None:
      files = self.database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'probe', **self.all_files_options)
    else:
      files = self.database.objects(protocol = self.protocol, groups = group, purposes = 'probe', **self.all_files_options)
    return self.sort(files)


  def probe_file_sets(self, model_id = None, group = 'dev'):
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

    files : [:py:class:`FileSet`] or something similar
      The list of file sets used to probe the model with the given model id."""
    if model_id is not None:
      file_sets = self.database.object_sets(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'probe', **self.all_files_options)
    else:
      file_sets = self.database.object_sets(protocol = self.protocol, groups = group, purposes = 'probe', **self.all_files_options)
    return self.sort(file_sets)


  def annotations(self, file):
    """annotations(file) -> annots

    Returns the annotations for the given File object, if available.

    **Parameters:**

    file : :py:class:`bob.db.verification.utils.File`
      The file for which annotations should be returned.

    **Returns:**

    annots : dict or None
      The annotations for the file, if available.
    """
    return self.database.annotations(file)


  def original_file_names(self, files):
    """original_file_names(files) -> paths

    Returns the full path of the original data of the given File objects, as returned by :py:meth:`bob.db.verification.utils.Database.original_file_names`.

    **Parameters:**

    files : [:py:class:`bob.db.verification.utils.File`]
      The list of file object to retrieve the original data file names for.

    **Returns:**

    paths : [str]
      The paths extracted for the files, in the same order.
    """
    return self.database.original_file_names(files, self.check_existence)



class DatabaseBobZT (DatabaseBob, DatabaseZT):
  """This class can be used whenever you have a database that follows the Bob ZT-norm verification database interface, which is defined in :py:class:`bob.db.verification.utils.ZTDatabase`.

  **Parameters:**

  database : derivative of :py:class:`bob.db.verification.utils.ZTDatabase`
    The database instance (such as a :py:class:`bob.db.mobio.Database`) that provides the actual interface, see :ref:`verification_databases` for a list.

  z_probe_options : dict
    Dictionary of options passed to the :py:meth:`bob.db.verification.utils.ZTDatabase.z_probe_files` database query when retrieving files for Z-probing.

  kwargs : ``key=value`` pairs
    The arguments of the :py:class:`DatabaseBob` base class constructor.

    .. note:: Usually, the ``name``, ``protocol``, ``training_depends_on_protocol`` and ``models_depend_on_protocol`` keyword parameters of the :py:class:`Database` base class constructor need to be specified.
  """

  def __init__(
      self,
      database,
      z_probe_options = {}, # Limit the z-probes
      **kwargs
  ):
#    assert isinstance(database, bob.db.verification.utils.ZTDatabase) // fails in tests
    # call base class constructor, passing all the parameters to it
    DatabaseBob.__init__(self, database = database, z_probe_options = z_probe_options, **kwargs)

    self.z_probe_options = z_probe_options


  def all_files(self, groups = ['dev']):
    """all_files(groups=None) -> files

    Returns all files of the database, including those for ZT norm, respecting the current protocol.
    The files can be limited using the ``all_files_options`` and the the ``z_probe_options`` in the constructor.

    **Parameters:**

    groups : some of ``('world', 'dev', 'eval')`` or ``None``
      The groups to get the data for.
      If ``None``, data for all groups is returned.

    **Returns:**

    files : [:py:class:`bob.db.verification.utils.File`]
      The sorted and unique list of all files of the database.
    """
    files = self.database.objects(protocol = self.protocol, groups = groups, **self.all_files_options)

    # add all files that belong to the ZT-norm
    for group in groups:
      if group == 'world': continue
      files += self.database.tobjects(protocol = self.protocol, groups = group, model_ids = None)
      files += self.database.zobjects(protocol = self.protocol, groups = group, **self.z_probe_options)
    return self.sort(files)


  def t_model_ids(self, group = 'dev'):
    """t_model_ids(group = 'dev') -> ids

    Returns a list of model ids of T-Norm models for the given group, respecting the current protocol.

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the model ids for.

    **Returns:**

    ids : [int] or [str]
      The list of (unique) model ids for T-Norm models of the given group.
    """
    return sorted(self.database.t_model_ids(protocol = self.protocol, groups = group))


  def t_enroll_files(self, t_model_id, group = 'dev'):
    """t_enroll_files(t_model_id, group = 'dev') -> files

    Returns a list of File objects that should be used to enroll the T-Norm model with the given model id from the given group, respecting the current protocol.

    **Parameters:**

    t_model_id : int or str
      A unique ID that identifies the model.

    group : one of ``('dev', 'eval')``
      The group to get the enrollment files for.

    **Returns:**

    files : [:py:class:`bob.db.verification.utils.File`]
      The sorted list of files used for to enroll the model with the given model id.
    """
    return self.sort(self.database.t_enroll_files(protocol = self.protocol, groups = group, model_id = t_model_id))


  def z_probe_files(self, group = 'dev'):
    """z_probe_files(group = 'dev') -> files

    Returns a list of probe files used to compute the Z-Norm, respecting the current protocol.
    The Z-probe files can be limited using the ``z_probe_options`` in the query to :py:meth:`bob.db.verification.utils.ZTDatabase.z_probe_files`

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the Z-norm probe files for.

    **Returns:**

    files : [:py:class:`bob.db.verification.utils.File`]
      The unique list of files used to compute the Z-norm.
    """
    files = self.database.z_probe_files(protocol = self.protocol, groups = group, **self.z_probe_options)
    return self.sort(files)


  def z_probe_file_sets(self, group = 'dev'):
    """z_probe_file_sets(group = 'dev') -> files

    Returns a list of probe FileSet objects used to compute the Z-Norm.
    The Z-probe files can be limited using the ``z_probe_options`` in the query to

    **Parameters:**

    group : one of ``('dev', 'eval')``
      The group to get the Z-norm probe files for.

    **Returns:**

    files : [:py:class:`FileSet`] or similar
      The unique list of file sets used to compute the Z-norm.
    """
    file_sets = self.database.z_probe_file_sets(protocol = self.protocol, groups = group, **self.z_probe_options)
    return self.sort(file_sets)
