import bob.db.bio_filelist
from .database import ZTBioDatabase
from .file import BioFile

class FileListBioDatabase (ZTBioDatabase):

  """This database is a gereric interface to use the :py:class:`bob.db.bio_filelist.Database` inside ``bob.bio``.
  Most of the constructor parameters stem from :py:class:`bob.db.bio_filelist.Database`, while some stem from :py:class:`BioDatabase`, particularly:
  
   * name
   * protocol (if your filelist database is not split into protocols, leave it blank)
   * training_depends_on_protocol
   * models_depend_on_protocol   
   * check_original_files_for_existence
   
  Annotation IO is actually handled by :py:class:`BioDatabase`, so if you have annotations, please specify:
  
   * annotation_directory
   * annotation_extension
   * annotation_type
  """

  def __init__(
    self,
    name,
    base_dir,
    protocol = None,
    dev_subdir = None,
    eval_subdir = None,
    world_filename = None,
    optional_world_1_filename = None,
    optional_world_2_filename = None,
    models_filename = None,
    # For probing, use ONE of the two score file lists:
    probes_filename = None,  # File containing the probe files -> dense model/probe score matrix
    scores_filename = None,  # File containing list of model and probe files -> sparse model/probe score matrix
    # For ZT-Norm:
    tnorm_filename = None,
    znorm_filename = None,
    use_dense_probe_file_list = None,
    # if both probe_filename and scores_filename is given, what kind of list should be used?
    keep_read_lists_in_memory = True,
    # additional parameters passed to the base class
    **kwargs
  ):
  
    # call base class constructor, storing all parameters
    super(FileListBioDatabase, self).__init__(
      name = name,
      base_dir = base_dir,
      protocol = protocol,
      dev_subdir = dev_subdir,
      eval_subdir = eval_subdir,
      world_filename = world_filename,
      optional_world_1_filename = optional_world_1_filename,
      optional_world_2_filename = optional_world_2_filename,
      models_filename = models_filename,
      probes_filename = probes_filename,
      scores_filename = scores_filename,
      tnorm_filename = tnorm_filename,
      znorm_filename = znorm_filename,
      use_dense_probe_file_list = use_dense_probe_file_list,
      # if both probe_filename and scores_filename is given, what kind of list should be used?
      keep_read_lists_in_memory = keep_read_lists_in_memory,
      **kwargs
    )
  
    # instantiate bio_filelist with given set of parameters  
    self.__db = bob.db.bio_filelist.Database(
      base_dir = base_dir,
      dev_subdir = dev_subdir,
      eval_subdir = eval_subdir,
      world_filename = world_filename,
      optional_world_1_filename = optional_world_1_filename,
      optional_world_2_filename = optional_world_2_filename,
      models_filename = models_filename,
      probes_filename = probes_filename,
      scores_filename = scores_filename,
      tnorm_filename = tnorm_filename,
      znorm_filename = znorm_filename,
      use_dense_probe_file_list = use_dense_probe_file_list,
      keep_read_lists_in_memory = keep_read_lists_in_memory
    )
    

  def _make_bio(self, files):
    return [BioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in files]


  def all_files(self, groups = ['dev']):
    files = self.__db.objects(self.protocol, None, None, groups, **self.all_files_options)
    # add all files that belong to the ZT-norm
    for group in groups:
      if group == 'world': continue
      if self.__db.implements_zt(self.protocol, group):
        files += self.__db.tobjects(self.protocol, None, group)
        files += self.__db.zobjects(self.protocol, group, **self.z_probe_options)
    return self.sort(self._make_bio(files))


  def model_ids(self, groups=None, protocol=None, **kwargs):
    return self.__db.model_ids(self.protocol, groups, **kwargs)

  def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
    return self.__db.tmodel_ids(protocol, groups, **kwargs)

  def client_id_from_model_id(self, model_id, group='dev'):
    return self.__db.get_client_id_from_model_id(model_id, group, self.protocol)

  def client_id_from_t_model_id(self, t_model_id, group='dev'):
    return self.__db.get_client_id_from_tmodel_id(t_model_id, group, self.protocol)

  def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
    return self._make_bio(self.__db.objects(protocol, purposes, model_ids, groups, **kwargs))

  def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
    return self._make_bio(self.__db.tobjects(protocol, model_ids, groups, **kwargs))

  def zobjects(self, groups=None, protocol=None, **kwargs):
    return self._make_bio(self.__db.zobjects(protocol, groups, **kwargs))

