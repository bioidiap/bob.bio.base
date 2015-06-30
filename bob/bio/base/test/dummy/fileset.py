import bob.db.atnt
import os

from bob.bio.base.database import DatabaseBob, DatabaseBobZT, File, FileSet
from bob.bio.base.test.utils import atnt_database_directory

class FileSetDatabase (DatabaseBobZT):

  def __init__(self):
    # call base class constructor with useful parameters
    DatabaseBobZT.__init__(
        self,
        database = bob.db.atnt.Database(
            original_directory = atnt_database_directory(),
        ),
        name = 'test_fileset',
        check_original_files_for_existence = True,
        training_depends_on_protocol = False,
        models_depend_on_protocol = False
    )

  def uses_probe_file_sets(self):
    return True

  def probe_file_sets(self, model_id = None, group = 'dev'):
    """Returns the list of probe File objects (for the given model id, if given)."""
    files = self.arrange_by_client(self.sort(self.database.objects(protocol = None, groups = group, purposes = 'probe')))
    # arrange files by clients
    file_sets = []
    for client_files in files:
      # convert into our File objects (so that they are tested as well)
      our_files = [File(f.id, f.client_id, f.path) for f in client_files]
      # generate file set for each client
      file_set = FileSet(our_files[0].client_id, our_files)
      file_sets.append(file_set)
    return file_sets

  def all_files(self, groups = ['dev']):
    return DatabaseBob.all_files(self, groups)

  def t_model_ids(self, group = 'dev'):
    return self.model_ids(group)

  def t_enroll_files(self, t_model_id, group = 'dev'):
    return self.enroll_files(t_model_id, group)

  def z_probe_files(self, group = 'dev'):
    return self.probe_files(None, group)

  def z_probe_file_sets(self, group = 'dev'):
    return self.probe_file_sets(None, group)

database = FileSetDatabase()
